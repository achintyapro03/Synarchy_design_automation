#include <stdio.h>
#include <stdlib.h>   // Needed for qsort

#define MM_N 48
#define QSORT_N 2000
#define RANDMEM_N 10000

/***************** MATRIX MULTIPLY *****************/
void matrix_multiply(void) {
    static double A[MM_N][MM_N];
    static double B[MM_N][MM_N];
    static double C[MM_N][MM_N];

    // Combined initialization
    for (int i = 0; i < MM_N; i++)
        for (int j = 0; j < MM_N; j++)
            A[i][j] = 1.0, B[i][j] = 2.0, C[i][j] = 0.0;

    // Multiply
    for (int i = 0; i < MM_N; i++)
        for (int k = 0; k < MM_N; k++)
            for (int j = 0; j < MM_N; j++)
                C[i][j] += A[i][k] * B[k][j];

    printf("Matrix multiply done\n");
}

/***************** QSORT *****************/
// Prototype for compare function
int compare_int(const void *a, const void *b);

void do_qsort(void) {
    static int arr[QSORT_N];

    // Fill with pseudo-random numbers
    int seed = 1;
    for (int i = 0; i < QSORT_N; i++)
        arr[i] = (seed = (seed * 1103515245 + 12345) & 0x7fffffff) % QSORT_N;

    qsort(arr, QSORT_N, sizeof(int), compare_int);

    printf("Qsort done\n");
}

int compare_int(const void *a, const void *b) {
    int x = *(const int*)a;
    int y = *(const int*)b;
    return (x > y) - (x < y);
}

/***************** RANDOM MEMORY ACCESS *****************/
void random_memory_access(void) {
    static int arr[RANDMEM_N];
    int seed = 1;

    // Fill with pseudo-random numbers
    for (int i = 0; i < RANDMEM_N; i++)
        arr[i] = (seed = (seed * 1103515245 + 12345) & 0x7fffffff) % RANDMEM_N;

    volatile int sum = 0;
    for (int i = 0; i < RANDMEM_N; i++)
        sum += arr[(seed = (seed * 1103515245 + 12345) & 0x7fffffff) % RANDMEM_N];

    printf("Random memory access done (sum=%d)\n", sum);
}

/***************** MAIN *****************/
int main(void) {
    printf("Starting benchmarks...\n");

    matrix_multiply();
    do_qsort();
    random_memory_access();

    printf("All benchmarks done\n");
    return 0;
}
