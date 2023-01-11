#ifndef GOLDEN
#define GOLDEN

void goldenSeq(float* images, float* filter, float* out,
               const int N, const int P, const int Q, const int K, const int C, const int R, const int S) {
    for (int n = 0; n < N; ++n) {
    for (int p = 0; p < P; ++p) {
    for (int q = 0; q < Q; ++q) {
    for (int k = 0; k < K; ++k) {
        float acc = 0.0f;
        for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
        for (int s = 0; s < S; ++s) {
            acc += images[n * (2 * P + R - 1) * (2 * Q + S - 1) * C +
                          (2 * p + r) * (2 * Q + S - 1) * C +
                          (2 * q + s) * C +
                          c] *
                   filter[k * R * S * C + r * S * C + s * C + c];
        }}}
        out[n * P * Q * K + p * Q * K + q * K + k] = acc;
    }}}}
}

#endif