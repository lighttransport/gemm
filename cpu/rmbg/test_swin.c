/* Independent double-precision checks for GEMM, LN and masked attention. */
#define SWIN_NO_MAIN
#include "../../common/swin_runner.c"

static double attention_reference(const float *qkv, const float *bias, int win,
                                  int head, int qi, int d, int shifted)
{
    double logits[144], maximum = -INFINITY, denominator = 0, numerator = 0;
    int c = 64, heads = 2, wy = win/2*12, wx = win%2*12;
    for (int k = 0; k < 144; k++) {
        double score = 0;
        for (int j = 0; j < 32; j++)
            score += (double)qkv[((size_t)win*144+qi)*3*c+head*32+j] *
                     qkv[((size_t)win*144+k)*3*c+c+head*32+j];
        score = score/sqrt(32.0)+bias[((qi/12-k/12+11)*23+qi%12-k%12+11)*heads+head];
        /* Explicit boundaries for a 24x24 padded grid; no production helper. */
        int qy = wy+qi/12, qx = wx+qi%12, ky = wy+k/12, kx = wx+k%12;
        int qry = (qy >= 12)+(qy >= 18), qrx = (qx >= 12)+(qx >= 18);
        int kry = (ky >= 12)+(ky >= 18), krx = (kx >= 12)+(kx >= 18);
        if (shifted && (qry != kry || qrx != krx)) score -= 100;
        logits[k] = score;
        if (score > maximum) maximum = score;
    }
    for (int k = 0; k < 144; k++) {
        double p = exp(logits[k]-maximum);
        denominator += p;
        numerator += p*qkv[((size_t)win*144+k)*3*c+2*c+head*32+d];
    }
    return numerator/denominator;
}

int main(int argc, char **argv)
{
    omp_set_num_threads(4);
#ifdef SWIN_CUDA
    if (argc == 2 && !strcmp(argv[1], "--cuda")) {
        swin_gpu = 1;
        if (cuda_linear_f32_init(0)) return 2;
    } else if (argc != 1) return 2;
#else
    (void)argv;
    if (argc != 1) return 2;
#endif
    float x[7*11], w[9*11], b[9], y[7*9];
    for (int i = 0; i < 7*11; i++) x[i] = sinf(i*.23f);
    for (int i = 0; i < 9*11; i++) w[i] = cosf(i*.11f);
    for (int i = 0; i < 9; i++) b[i] = i*.03f;
    swin_linear(y, w, b, x, 7, 9, 11);
    double error = 0;
    for (int i = 0; i < 7; i++) for (int j = 0; j < 9; j++) {
        double ref = b[j];
        for (int k = 0; k < 11; k++) ref += (double)x[i*11+k]*w[j*11+k];
        error = fmax(error, fabs(ref-y[i*9+j]));
    }
    printf("linear max_abs=%.9g\n", error);
    if (error > 2e-6) return 1;
    /* Odd long K tests panel tails, cancellation and multiple output tiles. */
    int lm = 5, ln = 67, lk = 779;
    float *lx = swin_alloc((size_t)lm*lk*sizeof(float));
    float *lw = swin_alloc((size_t)ln*lk*sizeof(float));
    float *ly = swin_alloc((size_t)lm*ln*sizeof(float));
    for (int i = 0; i < lm*lk; i++) lx[i] = sinf(i*.23f);
    for (int i = 0; i < ln*lk; i++) lw[i] = cosf(i*.11f);
    swin_linear(ly, lw, NULL, lx, lm, ln, lk);
    error = 0;
    for (int i = 0; i < lm; i++) for (int j = 0; j < ln; j++) {
        double ref = 0;
        for (int d = 0; d < lk; d++) ref += (double)lx[i*lk+d]*lw[j*lk+d];
        error = fmax(error, fabs(ref-ly[i*ln+j]));
    }
    free(lx); free(lw); free(ly);
    printf("long_linear max_abs=%.9g\n", error);
    if (error > 2e-5) return 1;
    float norm[7*11], nw[11], nb[11];
    for (int j = 0; j < 11; j++) { nw[j] = 1+j*.01f; nb[j] = j*.02f; }
    swin_norm(norm, x, nw, nb, 7, 11);
    error = 0;
    for (int i = 0; i < 7; i++) {
        double mean = 0, variance = 0;
        for (int j = 0; j < 11; j++) mean += x[i*11+j]/11.0;
        for (int j = 0; j < 11; j++) variance += pow(x[i*11+j]-mean, 2)/11.0;
        for (int j = 0; j < 11; j++) {
            double ref = (x[i*11+j]-mean)/sqrt(variance+1e-5)*nw[j]+nb[j];
            error = fmax(error, fabs(ref-norm[i*11+j]));
        }
    }
    printf("layer_norm max_abs=%.9g\n", error);
    if (error > 2e-6) return 1;
    float *qkv = swin_alloc(4*144*3*64*sizeof(float));
    float *out = swin_alloc(4*144*64*sizeof(float)), bias[529*2];
    for (int i = 0; i < 4*144*3*64; i++) qkv[i] = sinf(i*.017f)*.7f;
    for (int i = 0; i < 529*2; i++) bias[i] = cosf(i*.13f)*.3f;
    for (int shift = 0; shift <= 6; shift += 6) {
        swin_attention(out, qkv, bias, 24, 24, 64, shift);
        error = 0;
        for (int win = 0; win < 4; win++) for (int head = 0; head < 2; head++)
            for (int qi = 0; qi < 144; qi += 5) for (int d = 0; d < 32; d += 7) {
                double ref = attention_reference(qkv, bias, win, head, qi, d, shift);
                error = fmax(error, fabs(ref-out[((size_t)win*144+qi)*64+head*32+d]));
            }
        printf("attention shift=%d max_abs=%.9g\n", shift, error);
        if (error > 2e-6) return 1;
    }
    free(qkv); free(out);
#ifdef SWIN_CUDA
    cuda_linear_f32_free();
#endif
    puts("PASS");
    return 0;
}
