#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <pthread.h>
#include <unistd.h>
#include <stdatomic.h>

//gcc -O2 -march=native matmul.c -o c

#define N 2048
float A[N*N] __attribute__ ((aligned (64)));
float B[N*N] __attribute__ ((aligned (64)));
float C[N*N] __attribute__ ((aligned (64)));
float ans[N*N] __attribute__ ((aligned (64)));

__m256 *Am = (__m256*)A;
__m256 *Bm = (__m256*)B;
__m256 *Cm = (__m256*)C;

float Bf[N*N] __attribute__ ((aligned (64))); //no idea why geohot does this?
__m256 *Bfm = (__m256*)Bf;

void matmulAvx() {
      for(int y = 0; y < N; y++) {
         for(int k = 0; k < N; k++) {
            __m256 ta = _mm256_broadcast_ss(&A[(y*N) + k]);
            for(int x = 0; x < N; x+=8) {
               Cm[(y*N + x)/8] = _mm256_fmadd_ps(ta, Bm[((k*N) + x)/8], Cm[(y*N + x)/8]);
            }
         }
      }
   }

int main() {
   // printf() displays the string inside quotation
   printf("Hello, World!\n");


   float max =1;

   for(int i = 0; i < N*N; i++) {
       A[i] = (float)rand()/(float)(RAND_MAX/max);
       B[i] = (float)rand()/(float)(RAND_MAX/max);
       C[i] = 0;
       ans[i] = 0;
      //printf("%f %f %f %f\n",aa[i],bb[i],cc[i],an[i]);
   }
   
   clock_t begin = clock();
   for(int y = 0; y < N; y++) {
      for(int k = 0; k < N; k++) {
         for(int x = 0; x < N; x++) {
            ans[y*N + x] += A[y*N + k] * B[x + k*N];
         }
      }
   }
   clock_t end = clock();
   double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
   printf("reordered time spent = %f\n",time_spent);
   
   printf("DONE\n");

   int BLOCK = 8;
   begin = clock();

   matmulAvx();

   end = clock();
   time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
   printf("reordered + avx time spent = %f\n",time_spent);

   for(int i = 0; i < N*N; i++) {
      //printf("FFS %d %f -> %f\n",i,C[i],ans[i]);
      if(C[i] != ans[i]) {
         printf("\nWRONG avx ! %f -> %f\n",C[i],ans[i]);
         return 0;
      }
   } 

   printf("\nDONE avx\n");
   return 0;
}

void matmul() {
   
   for(int y = 0; y < N; y++) {
      for(int k = 0; k < N; k++) {
         for(int x = 0; x < N; x++) {
            C[y*N + x] += A[y*N + k] * B[x + (N * k)];
         }
      }
   }
   
   /*
   for(int y = 0; y < N; y++) {
      for(int x = 0; x < N; x++) {
         for(int k = 0; k < N; k++) {
            C[y*N + x] += A[y*N + k] * B[x + k*N];
         }
      }
   }
   */
   
}
