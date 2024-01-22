#include "stdio.h"
#include "time.h"
#include "stdlib.h"

#include "ops.h"

#define N (3000)
#define num_tests (100)

float A[N * N], B[N], C[N], D[N];
double times[num_tests];

int main () {
    srand(0);

    for (int t = 0; t < num_tests; t++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[N * i + j] = (float)rand()/((float)RAND_MAX);
            }
        }
        for (int i = 0; i < N; i++) {
            B[i] = (float)rand()/((float)RAND_MAX);
        }

        clock_t start = clock();
        matmul(A, B, C, N);
        softmax(C, D, N);
        clock_t end = clock();
        times[t] = ((double) (end - start)) / CLOCKS_PER_SEC;
    }

    double total_time = 0;
    for (int i = 0; i < num_tests; i++) total_time += times[i];
    printf("AVERAGE TIME: %f\n", total_time / num_tests);
}