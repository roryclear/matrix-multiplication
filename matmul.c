#include <stdio.h>
#include <stdlib.h>

#define N 2048
float A[N*N] __attribute__ ((aligned (64)));
float B[N*N] __attribute__ ((aligned (64)));
float C[N*N] __attribute__ ((aligned (64)));

int main() {
   // printf() displays the string inside quotation
   printf("Hello, World!");

   for(int i = 0; i < N*N; i++) {
      float a = 1;
      A[i] = (float)rand()/(float)(RAND_MAX/a);
      printf("%f\n",A[i]);
   }

   printf("hello %f",A[0]);
   return 0;
}