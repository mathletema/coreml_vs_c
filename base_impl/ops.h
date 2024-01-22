#ifndef OPS_H
#define OPS_H

#include "string.h"
#include "math.h"

// multiplies a matrix A of size NxN with a vector B of size N
void matmul(const float * restrict A, const float * restrict B, float * restrict C, const int N) {
    memset(C, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i] += A[N * i + j] * B[j];
        }
    }
}


// computes softmax of A
void softmax(const float * restrict A, float * restrict B, const int N) {
    float max_A = A[0];
    for (int i = 1; i < N; i++) {
        if (A[i] < max_A) max_A = A[i];
    }

    for (int i = 0; i < N; i++) {
        B[i] = exp(A[i] - max_A);
    }

    float sum = 0;
    for (int i = 0; i < N; i++) sum += B[i];
    
    for (int i = 0; i < N; i++) {
        B[i] /= sum;
    }
}


#endif