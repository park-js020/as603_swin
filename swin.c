#include <stdio.h>
#include <string.h>
#include <stdlib.h>


// Model Hyperparameters for ImageNet-1K Swin-Tiny
#define N_PATCH     3136      // 224x224 image / 4x4 patch = 56x56 patches
#define D_PATCH     48        // patch_size * patch_size * in_chans = 4*4*3
#define D_MODEL     96        // embedding dimension of stage 0 (Swin-Tiny: 96)
#define N_HEAD      3         // attention heads
#define D_FFN       384       // D_MODEL * 4 (FFN expansion ratio)
#define DEPTH       2         // depth of first stage (use per-stage depths later)
#define NUM_CLASSES 1000      // ImageNet-1K classes
#define D_HEAD      (D_MODEL / N_HEAD)  // 96/3 = 32


// Quantization Level
#define Q           20      // Q12.20 fixed-point, scale = 2^20
#define SCALE       (1 << Q)
#define MIN_INT     (-0x7fffffff - 1)
#define MAX_INT      0x7fffffff

// Swin-specific
#define PATCH_GRID_SIDE   (int_sqrt(N_PATCH))   // will evaluate at compile-time to 8 if possible
#define WINDOW_SIZE       7                     // window size (must divide PATCH_GRID_SIDE)
#define SHIFT_SIZE        (WINDOW_SIZE/2)       // shifted window shift

// Quantized Types
typedef int         INT32;
typedef long long   INT64;

typedef struct {
    int idx;
    float val;
} TopK;

// Compare function for qsort (descending)
int cmp_desc(const void *a, const void *b) {
    float diff = ((TopK*)b)->val - ((TopK*)a)->val;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}

// Load Weights from Header File
#include "swin_weights.h"    
#include "data_input.h"

/* ================================================================================================================== */

static inline INT32
itoq(int v) { return (INT32)(v << Q); }

static inline int 
qtoi(INT32 v) { return v >> Q; }

static inline INT32 
ftoq(float f) { return (INT32)(f * (float)SCALE); }

static inline float 
qtof(INT32 v) { return (float)v / (float)SCALE; }

/* ================================================================================================================== */

static inline int /* Square root of an ordinary integer */
int_sqrt(int x) {
    if (x <= 0) return 0;
    int r = x;
    int y = (r + 1) / 2;
    while (y < r) {
        r = y;
        y = (r + x / r) / 2;
    }
    return r;
}

/* ================================================================================================================== */

static inline INT32 /* Inverse of a quantized value */
inv_q(INT32 x) {
    if (x == 0) return MAX_INT;
    INT64 numerator = (INT64)1 << (2 * Q);
    return (INT32)(numerator / x);
}

static inline INT32 /* Multiplication of two quantized values */
mul_q(INT32 a, INT32 b) {
    INT64 temp = (INT64)a * (INT64)b; 
    INT64 rounded = (temp + (1LL << (Q - 1))) >> Q; 
    if (rounded > MAX_INT) return MAX_INT;
    if (rounded < MIN_INT) return MIN_INT;
    return (INT32)rounded;
}

static inline INT32 /* Fixed-point division: a / b */
div_q(INT32 a, INT32 b) {
    if (b == 0) return (a >= 0 ? MAX_INT : MIN_INT);
    return (INT32)(((INT64)a << Q) / b);
}

static inline INT32 /* Fixed-point square root: sqrt(x) where x is Q-format */
sqrt_q(INT32 x) {
    if (x <= 0) return 0;
    
    INT64 temp = (INT64)x << Q; 
    INT32 root = int_sqrt(temp);
    
    // Newton-Raphson method for integer square root.
    for (int i = 0; i < 10; i++) { 
        if (root == 0) return 0; 
        root = (root + (temp / root)) / 2;
    }
    
    // Return the final result. The value is already in the correct fixed-point format.
    return root;
}

static inline INT32 /* Exponent of quantized value (truncated Taylor) */
exp_q(INT32 x) {
    // clamp input to [-8, 8]
    if (x > itoq(8)) x  = itoq(8);
    if (x < itoq(-8)) x = itoq(-8);

    INT32 x2 = mul_q(x, x);
    INT32 x3 = mul_q(x2, x);
    INT32 x4 = mul_q(x3, x);

    INT32 one = itoq(1);
    INT32 res = one;
    res = res + x;                          // + x
    res = res + div_q(x2, itoq(2));         // + x^2/2
    res = res + div_q(x3, itoq(6));         // + x^3/6
    res = res + div_q(x4, itoq(24));        // + x^4/24
    return res;
}

/* ================================================================================================================== */

void /* In-place matrix addition */
matadd_(
    INT32 input1[], INT32 input2[], int d_in
) {
    for (int i = 0; i < d_in; i++) {
        input1[i] += input2[i];
    }
}

void /* Out-of-place matrix addition */
matadd(
    INT32 input1[], INT32 input2[], INT32 output[], INT32 d_in
) {
    for (int i = 0; i < d_in; i++) {
        output[i] = input1[i] + input2[i];
    }
}

void /* Matrix multiplication: mat1 is (n_tok * d_in), mat2 is (d_in * d_out), out is (n_tok * d_out) */
matmul(
    INT32 mat1[], INT32 mat2[], INT32 output[],
    int n_tok, int d_in, int d_out
) {
    for (int t = 0; t < n_tok; t++) {
        for (int i = 0; i < d_out; i++) {
            INT64 sum = 0;
            for (int j = 0; j < d_in; j++) {
                sum += (INT64)mul_q(mat1[t * d_in + j], mat2[j * d_out + i]);
            }
            /* bias addition isn't done here; caller adds bias */
            output[t * d_out + i] = (INT32)sum;
        }
    }
}

void /* Fully-connected layer with weight and bias */ 
linear(
    INT32 input[], INT32 weight[], INT32 bias[], INT32 output[],
    int n_tok, int d_in, int d_out
) {
    for (int t = 0; t < n_tok; t++) {           
        for (int i = 0; i < d_out; i++) {     
            INT32 sum = 0;  
            for (int j = 0; j < d_in; j++) {  
                sum += mul_q(input[t * d_in + j], weight[j * d_out + i]);
            }
            output[t * d_out + i] = (INT32)sum + bias[i];
        }
    }
}


/* ================================================================================================================== */

int /* Argmax */
argmax(INT32 input[], int input_size) {
    int max_idx = 0; 
    for (int i = 1; i < input_size; i++) {
        if (input[i] > input[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

void /* ReLU Activation */
relu(INT32 input[], int d_in) {
    for (int i = 0; i < d_in; i++) {
        input[i] = (input[i] < 0) ? 0 : input[i];
    }
}

/* ================================================================================================================== */

void /* Layer Normalization */
layer_norm(
    INT32 input[], INT32 output[],
    INT32 gamma[], INT32 beta[],
    int d 
) {
    INT64 sum = 0;
    for (int i = 0; i < d; i++) {
        sum += input[i];
    }
    INT32 mean = (INT32)(sum / d);
    
    INT64 var_sum = 0;
    for (int i = 0; i < d; i++) {
        INT32 diff = input[i] - mean;
        var_sum += (INT64)diff * diff;
    }
    INT32 var = (INT32)(var_sum / d);

    int stdev = sqrt_q(var);
    if (stdev == 0) stdev = 1;
    INT32 inv_stdev = inv_q(stdev);

    for (int i = 0; i < d; i++) {
        INT32 norm = mul_q((input[i] - mean), inv_stdev);
        output[i] = mul_q(norm, gamma[i]) + beta[i];
    }
}

void /* Rowwise Softmax */
softmax(INT32 score[], INT32 out_probs[], int n_tok) {
    for (int i = 0; i < n_tok; i++) {
        int row = i * n_tok;

        INT32 row_max = score[row];
        for (int j = 1; j < n_tok; j++) 
            if (score[row + j] > row_max) 
                row_max = score[row + j];

        INT64 sum_exp = 0;
        INT32 exps[n_tok];
        for (int j = 0; j < n_tok; j++) {
            INT32 x = score[row + j] - row_max;
            exps[j] = exp_q(x);
            sum_exp += exps[j];
        }

        if (sum_exp == 0) sum_exp = 1;

        for (int j = 0; j < n_tok; j++) {
            out_probs[row + j] = div_q(exps[j], (INT32)sum_exp);
        }
    }
}

/* ================================================================================================================== */
// Helpers for Swin-style window and shift

/* map sequence index -> 2D coords and back
   assume sequence is row-major: idx = r * W + c
*/
static inline void seq_to_rc(int idx, int W, int *r, int *c) {
    *r = idx / W;
    *c = idx % W;
}

static inline int rc_to_seq(int r, int c, int W) {
    return r * W + c;
}

/* cyclically shift a grid index (r,c) by shift in both directions */
static inline void shift_coord(int r, int c, int H, int W, int shift, int *r2, int *c2) {
    int rr = (r + shift) % H; if (rr < 0) rr += H;
    int cc = (c + shift) % W; if (cc < 0) cc += W;
    *r2 = rr; *c2 = cc;
}

/* partition windows: for each window top-left (r0,c0) step window_size */
static inline int n_windows_for(int H, int W, int window_size) {
    return (H / window_size) * (W / window_size);
}


/* For relative position bias handling */
static inline int rel_index_from_coords(int dr, int dc, int win) {
    int coords = (dr + (win - 1)) * (2 * win - 1) + (dc + (win - 1));
    return coords;
}

/* Patch merging (for downsample) 
   input: in_tok * C  arranged row-major (H * W = n_tok), H and W are even
   reduction_weight: INT32 matrix of shape (in_dim * out_dim) where in_dim = 4*C, out_dim = 2*C
   reduction_bias: optional bias of length out_dim (use NULL if absent)
   output: n_tok/4 * (out_dim)  (each 2x2 block becomes one token)
*/
void patch_merge(
    INT32 input[], INT32 output[],
    INT32 reduction_weight[], INT32 reduction_bias[], 
    int H, int W, int C /* input channels per token */
) {
    int out_H = H / 2;
    int out_W = W / 2;
    int out_C = C * 2;          // timm's PatchMerging: out_channels = 2*C
    int in_block = 4 * C;       // concat 4 tokens

    INT32 concat[4 * C];

    for (int r = 0; r < out_H; r++) {
        for (int c = 0; c < out_W; c++) {
            int out_idx = r * out_W + c;
            
            /* gather 2x2 block: (2r,2c),(2r,2c+1),(2r+1,2c),(2r+1,2c+1) */
            int patch_coords[4][2] = {
                {2*r,   2*c},
                {2*r,   2*c+1},
                {2*r+1, 2*c},
                {2*r+1, 2*c+1}
            };

            /* concat */
            for (int blk = 0; blk < 4; blk++) {
                int rr = patch_coords[blk][0];
                int cc = patch_coords[blk][1];
                int seq = rr * W + cc;
                for (int d = 0; d < C; d++) {
                    concat[blk * C + d] = input[seq * C + d];
                }
            }

            /* linear: concat (1 x 4C) * weight (4C * 2C) -> (1 x 2C) */
            for (int outc = 0; outc < out_C; outc++) {
                INT64 sum = 0;
                for (int k = 0; k < in_block; k++) {
                    sum += (INT64)mul_q(concat[k], reduction_weight[k * out_C + outc]);
                }
                INT32 val = (INT32)sum;
                if (reduction_bias) val += reduction_bias[outc];
                output[out_idx * out_C + outc] = val;
            }
        }
    }
}

/* ================================================================================================================== */
// modules for Swin attention and blocks
void /* Single-head attention */ 
attn(
    INT32 input[], INT32 output[],
    INT32 w_query[], INT32 w_key[], INT32 w_value[],
    INT32 rel_pos_bias_table[], int rel_table_size, int n_head, int head_idx,
    int n_tok, int d_model, int d_head,
    int H, int W, int window_size, int shift
) {
    int win = window_size;
    int n_wtok = win * win;
    if (H * W != n_tok) {
        attn(input, output, w_query, w_key, w_value, rel_pos_bias_table, rel_table_size, n_head, head_idx, n_tok, d_model, d_head, H, W, 0, 0);
        return;
    }

    int *rel_idx = (int*)malloc(n_wtok * n_wtok * sizeof(int));
    if (!rel_idx) { fprintf(stderr, "[Error] malloc rel_idx failed\n"); return; }
    for (int i = 0; i < n_wtok; i++) {
        int ri = i / win, ci = i % win;
        for (int j = 0; j < n_wtok; j++) {
            int rj = j / win, cj = j % win;
            int dr = ri - rj;
            int dc = ci - cj;
            rel_idx[i * n_wtok + j] = rel_index_from_coords(dr, dc, win);
        }
    }

    int **idx_shift = (int**)malloc(H * sizeof(int*));
    if (!idx_shift) { free(rel_idx); fprintf(stderr, "[Error] malloc idx_shift failed\n"); return; }
    for (int i = 0; i < H; i++) {
        idx_shift[i] = (int*)malloc(W * sizeof(int));
        if (!idx_shift[i]) {
            for (int j = 0; j < i; j++) free(idx_shift[j]);
            free(idx_shift); free(rel_idx);
            fprintf(stderr, "[Error] malloc idx_shift row failed\n");
            return;
        }
    }
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++) {
            int rs, cs;
            if (shift != 0) shift_coord(r, c, H, W, shift, &rs, &cs);
            else { rs = r; cs = c; }
            idx_shift[r][c] = rc_to_seq(rs, cs, W);
        }

    INT32 *win_input = (INT32*)malloc(n_wtok * d_model * sizeof(INT32));
    INT32 *win_query = (INT32*)malloc(n_wtok * d_head * sizeof(INT32));
    INT32 *win_key   = (INT32*)malloc(n_wtok * d_head * sizeof(INT32));
    INT32 *win_value = (INT32*)malloc(n_wtok * d_head * sizeof(INT32));
    INT32 *win_score = (INT32*)malloc(n_wtok * n_wtok * sizeof(INT32));
    INT32 *win_probs = (INT32*)malloc(n_wtok * n_wtok * sizeof(INT32));
    INT32 *win_out   = (INT32*)malloc(n_wtok * d_head * sizeof(INT32));

    // handle malloc fails
    if (!win_input || !win_query || !win_key || !win_value || !win_score || !win_probs || !win_out) {
        for (int i = 0; i < H; i++) free(idx_shift[i]);
        free(idx_shift); free(rel_idx);
        if (win_input) free(win_input); if (win_query) free(win_query); if (win_key) free(win_key);
        if (win_value) free(win_value); if (win_score) free(win_score); if (win_probs) free(win_probs); if (win_out) free(win_out);
        fprintf(stderr, "[Error] malloc win buffers failed\n");
        return;
    }

    // number of windows (row, column)
    int n_win_r = H / win;
    int n_win_c = W / win;
    
    // apply windowed attention
    for (int wr = 0; wr < n_win_r; wr++) {
        for (int wc = 0; wc < n_win_c; wc++) {
            int r0 = wr * win;
            int c0 = wc * win;
            /* gather tokens for this window (shifted) */
            for (int dr = 0; dr < win; dr++) {
                for (int dc = 0; dc < win; dc++) {
                    int rr = r0 + dr;
                    int cc = c0 + dc;
                    int seq_idx_shifted = idx_shift[rr][cc];
                    int win_pos = dr * win + dc;
                    for (int d = 0; d < d_model; d++)
                        win_input[win_pos * d_model + d] = input[seq_idx_shifted * d_model + d];
                }
            }

            // QKV
            matmul(win_input, w_query, win_query, n_wtok, d_model, d_head);
            matmul(win_input, w_key,   win_key,   n_wtok, d_model, d_head);
            matmul(win_input, win_value, win_value, n_wtok, d_model, d_head);

            for (int i = 0; i < n_wtok; i++) {
                for (int j = 0; j < n_wtok; j++) {
                    INT64 s = 0;
                    for (int k = 0; k < d_head; k++) {
                        s += (INT64)mul_q(win_query[i * d_head + k], win_key[j * d_head + k]);
                    }
                    if (rel_pos_bias_table != NULL) {
                        int ridx = rel_idx[i * n_wtok + j];
                        INT32 bias_q = rel_pos_bias_table[ridx * n_head + head_idx];
                        s += bias_q;
                    }
                    win_score[i * n_wtok + j] = (INT32)s;
                }
            }

            INT32 scale = sqrt_q(itoq(d_head));
            INT32 inv_sqrt_d = inv_q(scale);
            for (int i = 0; i < n_wtok * n_wtok; i++)
                win_score[i] = mul_q(win_score[i], inv_sqrt_d);

            // Softmax
            softmax(win_score, win_probs, n_wtok);

            // Score
            matmul(win_probs, win_value, win_out, n_wtok, n_wtok, d_head);

            // Resize the array to 1D array
            for (int dr = 0; dr < win; dr++) {
                for (int dc = 0; dc < win; dc++) {
                    int rr = r0 + dr;
                    int cc = c0 + dc;
                    int win_pos = dr * win + dc;
                    int r_orig = rr, c_orig = cc;
                    if (shift != 0) {
                        r_orig = (rr - shift) % H; if (r_orig < 0) r_orig += H;
                        c_orig = (cc - shift) % W; if (c_orig < 0) c_orig += W;
                    }
                    int seq_orig = rc_to_seq(r_orig, c_orig, W);
                    for (int k = 0; k < d_head; k++)
                        output[seq_orig * d_head + k] = win_out[win_pos * d_head + k];
                }
            }
        }
    }
    // free heap
    for (int i = 0; i < H; i++) free(idx_shift[i]);

    // free malloc-ed buffers. 
    free(idx_shift); free(rel_idx);
    free(win_input); free(win_query); free(win_key); free(win_value);
    free(win_score); free(win_probs); free(win_out);
}

void /* Multi-head attention */
mha(
    INT32 input[], INT32 output[], 
    INT32 w_query[], INT32 w_key[], INT32 w_value[],
    INT32 w_out[], INT32 b_out[],
    INT32 rel_pos_bias_table[], int rel_table_size,  // new args
    int n_tok, int n_head, int d_model,
    int H, int W, int window_size, int shift
) {
    int d_head = d_model / n_head;
    INT32 *concat_score = (INT32*)malloc(n_tok * d_model * sizeof(INT32));
    if (!concat_score) { fprintf(stderr, "[Error] malloc concat_score failed\n"); return; }

    for (int h = 0; h < n_head; h++) {
        /* each head's projection matrices are arranged as consecutive d_model*d_head blocks */
        INT32 *wq_h = &w_query[h * d_model * d_head];
        INT32 *wk_h = &w_key[h * d_model * d_head];
        INT32 *wv_h = &w_value[h * d_model * d_head];

        INT32 *head_score = (INT32*)malloc(n_tok * d_head * sizeof(INT32));
        if (!head_score) { free(concat_score); fprintf(stderr, "[Error] malloc head_score failed\n"); return; }
        attn(
            input, head_score,
            wq_h, wk_h, wv_h,
            rel_pos_bias_table, rel_table_size, n_head, h,
            n_tok, d_model, d_head,
            H, W, window_size, shift
        );
        for (int t = 0; t < n_tok; t++) {
            for (int d = 0; d < d_head; d++) {
                concat_score[t * d_model + h * d_head + d] = head_score[t * d_head + d];
            }
        }
        free(head_score);
    }

    linear(concat_score, w_out, b_out, output, n_tok, d_model, d_model);
    free(concat_score);
}


void /* Feed forward network */
ffn(
    INT32 input[], INT32 output[], 
    INT32 w_up[], INT32 w_down[], 
    INT32 b_up[], INT32 b_down[], 
    int n_tok, int d_model, int d_ffn
) {
    int hidden_size = n_tok * d_ffn;
    INT32 *hidden = (INT32*)malloc(hidden_size * sizeof(INT32));
    if (!hidden) { fprintf(stderr, "[Error] malloc ffn hidden failed\n"); return; }
    linear(input, w_up, b_up, hidden, n_tok, d_model, d_ffn);
    relu(hidden, hidden_size);
    linear(hidden, w_down, b_down, output, n_tok, d_ffn, d_model);
    free(hidden);
}

/* ================================================================================================================== */
// block modules for Swin
void /* Swin embedding */
emb(
    INT32 input[], INT32 output[], 
    INT32 w_patch[], INT32 b_patch[], 
    INT32 cls_token[], INT32 pos_emb[], 
    int n_patch, int d_patch, int d_model
) {
    for (int j = 0; j < d_model; j++) {
        output[j] = cls_token[j] + pos_emb[j];
    }

    for (int p = 0; p < n_patch; p++) {
        INT32 *in_patch = &input[p * d_patch];
        INT32 *out_patch = &output[(p + 1) * d_model]; 
        INT32 *pos_patch = &pos_emb[(p + 1) * d_model];

        for (int i = 0; i < d_model; i++) {
            INT64 sum = 0;
            for (int k = 0; k < d_patch; k++) {
                sum += mul_q(w_patch[i * d_patch + k], in_patch[k]);
            }
            sum += b_patch[i];
            out_patch[i] = (INT32)sum + pos_patch[i];
        }
    }
}

/* Encoder block with shifted-window attention: 
   - case shift==0:             regular windowed attention
   - case shift==SHIFT_SIZE:    shifted-window attention
*/
void 
enc_blk(
    INT32 input[], INT32 output[], 
    INT32 gamma1[], INT32 beta1[],
    INT32 gamma2[], INT32 beta2[],
    INT32 w_query[], INT32 w_key[], INT32 w_value[],
    INT32 w_out[], INT32 b_out[],
    INT32 w_up[], INT32 b_up[], INT32 w_down[], INT32 b_down[],
    INT32 rel_pos_bias_table[], int rel_table_size,   // NEW
    int n_tok, int n_head, int d_model, int d_ffn,
    int H, int W, int window_size, int shift
) {
    int d = n_tok * d_model;
    INT32 *buffer_attn = (INT32*)malloc(d * sizeof(INT32));
    INT32 *buffer_ffn  = (INT32*)malloc(d * sizeof(INT32));
    if (!buffer_attn || !buffer_ffn) {
        if (buffer_attn) free(buffer_attn);
        if (buffer_ffn) free(buffer_ffn);
        fprintf(stderr, "[Error] malloc enc_blk buffers failed\n");
        return;
    }

    /* LayerNorm then MHA */
    layer_norm(input, buffer_attn, gamma1, beta1, d);
    mha(buffer_attn, output,
        w_query, w_key, w_value,
        w_out, b_out,
        rel_pos_bias_table, rel_table_size,
        n_tok, n_head, d_model, H, W, window_size, shift);
        
    /* Residual */
    matadd_(output, buffer_attn, d);
    
    /* Feed-forward */
    layer_norm(output, buffer_ffn, gamma2, beta2, d);
    ffn(buffer_ffn, output, w_up, w_down, b_up, b_down, n_tok, d_model, d_ffn);
    matadd_(output, buffer_ffn, d);
    free(buffer_attn); free(buffer_ffn);
}

/* ================================================================================================================== */
int /* Full swin-tiny implementation */ 
swin(INT32 input[]) {

    const int STAGE_DEPTHS[4] = {2, 2, 6, 2};
    const int NUM_STAGES = 4;

    const int STAGE_CHS[4] = {96, 192, 384, 768};

    // (1) Patch embedding -> produce tokens of dim STAGE_CHS[0]
    int H = int_sqrt(N_PATCH);
    int W = H;
    int n_tok = N_PATCH; // H*W
    int cur_dim = STAGE_CHS[0];

    INT32 *tokens_buf1 = (INT32*)malloc(N_PATCH * STAGE_CHS[3] * sizeof(INT32));
    INT32 *tokens_buf2 = (INT32*)malloc(N_PATCH * STAGE_CHS[3] * sizeof(INT32));
    if (!tokens_buf1 || !tokens_buf2) {
        fprintf(stderr, "[Error] Failed to allocate token buffers.\n");
        if (tokens_buf1) free(tokens_buf1);
        if (tokens_buf2) free(tokens_buf2);
        return -1;
    }

    INT32 *tokens_in = tokens_buf1;
    INT32 *tokens_out = tokens_buf2;

    /* Patch generation */ 
    for (int p = 0; p < n_tok; p++) {
        INT32 *in_patch  = &input[p * D_PATCH];    // input patches flattened
        INT32 *out_token = &tokens_in[p * cur_dim];
        for (int i = 0; i < cur_dim; i++) {
            INT64 sum = 0;
            for (int k = 0; k < D_PATCH; k++) {
                sum += (INT64)mul_q(w_patch_swin[i * D_PATCH + k], in_patch[k]);
            }
            sum += b_patch_swin[i];
            out_token[i] = (INT32)sum;
        }
    }

    /* LN for Patch */
    for (int t = 0; t < n_tok; t++) {
        layer_norm(&tokens_in[t * cur_dim], &tokens_in[t * cur_dim],
            patch_embed_norm_weight, patch_embed_norm_bias, cur_dim);
        }
        
    /* Swin stages */
    for (int stage = 0; stage < NUM_STAGES; stage++) {
        int depth = STAGE_DEPTHS[stage];
        cur_dim = STAGE_CHS[stage];
        
        for (int b = 0; b < depth; b++) {
            int shift = (b % 2 == 1) ? SHIFT_SIZE : 0;
            int rel_table_size = (2 * WINDOW_SIZE - 1) * (2 * WINDOW_SIZE - 1);

            /* Swin stage 0 */
            if (stage == 0) {
                if (b == 0) {
                    enc_blk(tokens_in, tokens_out,
                        L0_B0__ln1_gamma, L0_B0__ln1_beta,
                        L0_B0__ln2_gamma, L0_B0__ln2_beta,
                        L0_B0__q_w, L0_B0__k_w, L0_B0__v_w,
                        L0_B0__proj_w, L0_B0__proj_b,
                        L0_B0__fc1_w, L0_B0__fc1_b, L0_B0__fc2_w, L0_B0__fc2_b,
                        L0_B0__rel_pos_bias_table, rel_table_size,
                        n_tok, N_HEAD, cur_dim, D_FFN,
                        H, W, WINDOW_SIZE, shift);
                } else { 
                    enc_blk(tokens_in, tokens_out,
                        L0_B1__ln1_gamma, L0_B1__ln1_beta,
                        L0_B1__ln2_gamma, L0_B1__ln2_beta,
                        L0_B1__q_w, L0_B1__k_w, L0_B1__v_w,
                        L0_B1__proj_w, L0_B1__proj_b,
                        L0_B1__fc1_w, L0_B1__fc1_b, L0_B1__fc2_w, L0_B1__fc2_b,
                        L0_B1__rel_pos_bias_table, rel_table_size,
                        n_tok, N_HEAD, cur_dim, D_FFN,
                        H, W, WINDOW_SIZE, shift);
                }
            }
             /* Swin stage 1 */
            else if (stage == 1) {
                if (b == 0) {
                    enc_blk(tokens_in, tokens_out,
                        L1_B0__ln1_gamma, L1_B0__ln1_beta,
                        L1_B0__ln2_gamma, L1_B0__ln2_beta,
                        L1_B0__q_w, L1_B0__k_w, L1_B0__v_w,
                        L1_B0__proj_w, L1_B0__proj_b,
                        L1_B0__fc1_w, L1_B0__fc1_b, L1_B0__fc2_w, L1_B0__fc2_b,
                        L1_B0__rel_pos_bias_table, rel_table_size,
                        n_tok, N_HEAD, cur_dim, D_FFN,
                        H, W, WINDOW_SIZE, shift);
                } else { // b==1
                    enc_blk(tokens_in, tokens_out,
                        L1_B1__ln1_gamma, L1_B1__ln1_beta,
                        L1_B1__ln2_gamma, L1_B1__ln2_beta,
                        L1_B1__q_w, L1_B1__k_w, L1_B1__v_w,
                        L1_B1__proj_w, L1_B1__proj_b,
                        L1_B1__fc1_w, L1_B1__fc1_b, L1_B1__fc2_w, L1_B1__fc2_b,
                        L1_B1__rel_pos_bias_table, rel_table_size,
                        n_tok, N_HEAD, cur_dim, D_FFN,
                        H, W, WINDOW_SIZE, shift);
                }
            }
             /* Swin stage 2 */
            else if (stage == 2) {
                // blocks 0 to 5
                if (b == 0) {
                    enc_blk(tokens_in, tokens_out,
                        L2_B0__ln1_gamma, L2_B0__ln1_beta,
                        L2_B0__ln2_gamma, L2_B0__ln2_beta,
                        L2_B0__q_w, L2_B0__k_w, L2_B0__v_w,
                        L2_B0__proj_w, L2_B0__proj_b,
                        L2_B0__fc1_w, L2_B0__fc1_b, L2_B0__fc2_w, L2_B0__fc2_b,
                        L2_B0__rel_pos_bias_table, rel_table_size,
                        n_tok, N_HEAD, cur_dim, D_FFN,
                        H, W, WINDOW_SIZE, shift);
                } else if (b == 1) {
                    enc_blk(tokens_in, tokens_out,
                        L2_B1__ln1_gamma, L2_B1__ln1_beta,
                        L2_B1__ln2_gamma, L2_B1__ln2_beta,
                        L2_B1__q_w, L2_B1__k_w, L2_B1__v_w,
                        L2_B1__proj_w, L2_B1__proj_b,
                        L2_B1__fc1_w, L2_B1__fc1_b, L2_B1__fc2_w, L2_B1__fc2_b,
                        L2_B1__rel_pos_bias_table, rel_table_size,
                        n_tok, N_HEAD, cur_dim, D_FFN,
                        H, W, WINDOW_SIZE, shift);
                } else if (b == 2) {
                    enc_blk(tokens_in, tokens_out,
                        L2_B2__ln1_gamma, L2_B2__ln1_beta,
                        L2_B2__ln2_gamma, L2_B2__ln2_beta,
                        L2_B2__q_w, L2_B2__k_w, L2_B2__v_w,
                        L2_B2__proj_w, L2_B2__proj_b,
                        L2_B2__fc1_w, L2_B2__fc1_b, L2_B2__fc2_w, L2_B2__fc2_b,
                        L2_B2__rel_pos_bias_table, rel_table_size,
                        n_tok, N_HEAD, cur_dim, D_FFN,
                        H, W, WINDOW_SIZE, shift);
                } else if (b == 3) {
                    enc_blk(tokens_in, tokens_out,
                        L2_B3__ln1_gamma, L2_B3__ln1_beta,
                        L2_B3__ln2_gamma, L2_B3__ln2_beta,
                        L2_B3__q_w, L2_B3__k_w, L2_B3__v_w,
                        L2_B3__proj_w, L2_B3__proj_b,
                        L2_B3__fc1_w, L2_B3__fc1_b, L2_B3__fc2_w, L2_B3__fc2_b,
                        L2_B3__rel_pos_bias_table, rel_table_size,
                        n_tok, N_HEAD, cur_dim, D_FFN,
                        H, W, WINDOW_SIZE, shift);
                } else if (b == 4) {
                    enc_blk(tokens_in, tokens_out,
                        L2_B4__ln1_gamma, L2_B4__ln1_beta,
                        L2_B4__ln2_gamma, L2_B4__ln2_beta,
                        L2_B4__q_w, L2_B4__k_w, L2_B4__v_w,
                        L2_B4__proj_w, L2_B4__proj_b,
                        L2_B4__fc1_w, L2_B4__fc1_b, L2_B4__fc2_w, L2_B4__fc2_b,
                        L2_B4__rel_pos_bias_table, rel_table_size,
                        n_tok, N_HEAD, cur_dim, D_FFN,
                        H, W, WINDOW_SIZE, shift);
                } else { // b==5
                    enc_blk(tokens_in, tokens_out,
                        L2_B5__ln1_gamma, L2_B5__ln1_beta,
                        L2_B5__ln2_gamma, L2_B5__ln2_beta,
                        L2_B5__q_w, L2_B5__k_w, L2_B5__v_w,
                        L2_B5__proj_w, L2_B5__proj_b,
                        L2_B5__fc1_w, L2_B5__fc1_b, L2_B5__fc2_w, L2_B5__fc2_b,
                        L2_B5__rel_pos_bias_table, rel_table_size,
                        n_tok, N_HEAD, cur_dim, D_FFN,
                        H, W, WINDOW_SIZE, shift);
                }
            }
             /* Swin stage 3 */
            else { 
                // blocks 0, 1
                if (b == 0) {
                    enc_blk(tokens_in, tokens_out,
                        L3_B0__ln1_gamma, L3_B0__ln1_beta,
                        L3_B0__ln2_gamma, L3_B0__ln2_beta,
                        L3_B0__q_w, L3_B0__k_w, L3_B0__v_w,
                        L3_B0__proj_w, L3_B0__proj_b,
                        L3_B0__fc1_w, L3_B0__fc1_b, L3_B0__fc2_w, L3_B0__fc2_b,
                        L3_B0__rel_pos_bias_table, rel_table_size,
                        n_tok, N_HEAD, cur_dim, D_FFN,
                        H, W, WINDOW_SIZE, shift);
                } else {
                    enc_blk(tokens_in, tokens_out,
                        L3_B1__ln1_gamma, L3_B1__ln1_beta,
                        L3_B1__ln2_gamma, L3_B1__ln2_beta,
                        L3_B1__q_w, L3_B1__k_w, L3_B1__v_w,
                        L3_B1__proj_w, L3_B1__proj_b,
                        L3_B1__fc1_w, L3_B1__fc1_b, L3_B1__fc2_w, L3_B1__fc2_b,
                        L3_B1__rel_pos_bias_table, rel_table_size,
                        n_tok, N_HEAD, cur_dim, D_FFN,
                        H, W, WINDOW_SIZE, shift);
                }
            }

            // swap buffers for next block (enc_blk wrote into tokens_out)
            INT32 *tmp = tokens_in;
            tokens_in = tokens_out;
            tokens_out = tmp;
        } // end blocks for this stage

        // 2.1 After stage: if not last stage, apply downsample (patch merging)
        if (stage < NUM_STAGES - 1) {
            int next_stage = stage + 1;
            int out_H = H / 2;
            int out_W = W / 2;
            int out_n_tok = out_H * out_W;
            int in_C = STAGE_CHS[stage];
            int out_C = STAGE_CHS[next_stage]; // should equal in_C * 2

            if (next_stage == 1) {
                patch_merge(tokens_in, tokens_out,
                    L1__downsample__reduction_w,
                    NULL, 
                    H, W, in_C);
                for (int t = 0; t < out_n_tok; t++) {
                    layer_norm(&tokens_out[t * out_C], &tokens_out[t * out_C],
                        L1__downsample__norm_w, L1__downsample__norm_b, out_C);
                }
            } else if (next_stage == 2) {
                patch_merge(tokens_in, tokens_out,
                    L2__downsample__reduction_w,
                    NULL,
                    H, W, in_C);
                for (int t = 0; t < out_n_tok; t++) {
                    layer_norm(&tokens_out[t * out_C], &tokens_out[t * out_C],
                        L2__downsample__norm_w, L2__downsample__norm_b, out_C);
                }
            } else { 
                patch_merge(tokens_in, tokens_out,
                    L3__downsample__reduction_w,
                    NULL,
                    H, W, in_C);
                for (int t = 0; t < out_n_tok; t++) {
                    layer_norm(&tokens_out[t * out_C], &tokens_out[t * out_C],
                        L3__downsample__norm_w, L3__downsample__norm_b, out_C);
                }
            }

            // swap pointers so tokens_in points to merged tokens
            H = out_H; W = out_W; n_tok = out_n_tok;
            INT32 *tmp2 = tokens_in;
            tokens_in = tokens_out;
            tokens_out = tmp2;
        } 
    } 

     /* FC Norm */
    int final_dim = STAGE_CHS[NUM_STAGES - 1]; // last stage channels
    for (int t = 0; t < n_tok; t++) {
        layer_norm(&tokens_in[t * final_dim], &tokens_in[t * final_dim],
                   final_norm_w, final_norm_b, final_dim);
    }

    /* Pooling */
    INT32 pooled[final_dim];
    for (int j = 0; j < final_dim; j++) {
        INT64 sum = 0;
        for (int t = 0; t < n_tok; t++) sum += tokens_in[t * final_dim + j];
        pooled[j] = (INT32)(sum / n_tok);
    }

     /* MLP Classifier */
    INT32 logits[NUM_CLASSES];
    linear(pooled, w_cls_swin, b_cls_swin, logits, 1, final_dim, NUM_CLASSES);

    // Prepare array for sorting top-5
    TopK topk[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        topk[i].idx = i;
        topk[i].val = qtof(logits[i]);
    }

    // Sort in descending order
    qsort(topk, NUM_CLASSES, sizeof(TopK), cmp_desc);

    // Print top 5 predictions
    printf("[Top 5 Predictions]:\n");
    for (int i = 0; i < 5; i++) {
        printf("Class %d: %.6f\n", topk[i].idx, topk[i].val);
    }

    // Return top-1 class index
    int result = topk[0].idx;

    free(tokens_buf1);
    free(tokens_buf2);
    return result;
}

/* ================================================================================================================== */
int main(void) {
    // The input_data array from data_input.h, has size N_PATCH * D_PATCH
    INT32 *input = input_data;

    // Call Swin-Tiny
    int pred = swin(input);

    printf("[Predicted Class]: %d\n", pred);
    return 0;
}
