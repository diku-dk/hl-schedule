#ifndef GOLDEN
#define GOLDEN

void goldenSeq(float* a, float* b, float* c, const int M, const int N, const int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += a[m * K + k] * b[k * N + n];
            }
            c[m * N + n] = acc;
        }
    }
}

#endif