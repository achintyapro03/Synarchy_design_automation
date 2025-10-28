#include <stdio.h>

#define N 64   // Use 32 or 64 for quick simulation

static double A[N][N];
static double B[N][N];
static double C[N][N];

int main() {
    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 2.0;
            C[i][j] = 0.0;
        }
    }

    // Matrix multiplication with progress indicator
    for (int i = 0; i < N; i++) {
        if (i % (N/4 == 0 ? 1 : N/4) == 0) {   // Print progress 4 times
            printf("Row %d of %d\n", i, N);
            fflush(stdout);
        }
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    printf("done %d\n", N);
    fflush(stdout);
    return 0;
}
