#include <iostream>
#include <random>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
//#include "/usr/local/opt/libomp/include/omp.h"

//g++ -O3 -march=native -ffast-math matmul.cpp -o a

//g++ -O3 -march=native -ffast-math -fopenmp matmul.cpp -o a

//g++ -O3 -ffast-math -fopenmp matmul.cpp -o a

// for mac: https://apple.stackexchange.com/questions/99077/how-to-set-gcc-4-8-as-default-gcc-compiler/99157#99157

inline void matmulImplNaive(const float *left, const float *right,
                            float *result, int dim) {
  for (int y = 0; y < dim; y++) {
    for (int x = 0; x < dim; x++) {
      for (int k = 0; k < dim; k++) {
        result[y * dim + x] +=
            left[y * dim + k] * right[k * dim + x];
      } 
    } 
  } 
}


inline void matmulFaster(const float *left, const float *right,
                            float *result, int dim) {
  for (int y = 0; y < dim; y++) {
    for (int k = 0; k < dim; k++) {
      float lnum = left[y * dim + k];
      for (int x = 0; x < dim; x++) {
        result[y * dim + x] +=
            lnum * right[k * dim + x];
      } 
    } 
  } 
}


inline void matmulTiling(const float *left, const float *right,
                            float *result, int dim) {
  int xs = 256;
  int ys = 16;
  int ks = 8;
  for(int yt = 0; yt < dim; yt+=ys) {
    for (int kt = 0; kt < dim; kt+=ks) {
      for(int xt = 0; xt < dim; xt+=xs) {
        for (int y = yt; y < yt+ys; y++) {
          for(int k = kt; k < kt+ks; k++) {
            for (int x = xt; x < xt+xs; x++) {
              result[y * dim + x] +=
                  left[y * dim + k] * right[k * dim + x];
            }
          }
        }
      }
    }
  }
}


//1024 4 4 best on xps
// 256 16 8 best on macbook
inline void matmulTilingMulti(const float *left, const float *right,
                            float *result, int dim) {
  int xs = 256;
  int ys = 16;
  int ks = 4;
  #pragma omp parallel for
  for(int yt = 0; yt < dim; yt+=ys) {
    for (int kt = 0; kt < dim; kt+=ks) {
      for(int xt = 0; xt < dim; xt+=xs) {
        for (int y = yt; y < yt+ys; y++) {
          for(int k = kt; k < kt+ks; k++) {
            for (int x = xt; x < xt+xs; x++) {
              result[y * dim + x] +=
                  left[y * dim + k] * right[k * dim + x];
            }
          }
        }
      }
    }
  }
}


union U256f {
    __m256 v;
    float a[8];
};

inline void print(const __m256 v)
{
    const U256f u = { v };

    for (int i = 0; i < 8; ++i)
        printf("%f\n",u.a[i]);
}

inline void matmulNew(const float *left, const float *right,
                            float *result, int dim) {
    __m256 *rightm = (__m256*)right;
    __m256 *resultm = (__m256*)result;

    #pragma omp parallel for
    for(int i = 0; i < dim*dim; i++) {
      __m256 lm = _mm256_broadcast_ss(&left[i]);
      for(int j = 0; j < dim; j+=8) {
        resultm[(((i / dim) * dim) + j) / 8] = _mm256_fmadd_ps(lm,rightm[(((i % dim) * dim) + j) / 8],resultm[(((i / dim) * dim) + j) / 8]);
      }
    }
}


///////////////////////////////////ABOVE

inline void matmulNew2(const float *left, const float *right,
                            float *result, int dim) {
    for(int i = 0; i < dim*dim; i++) {
      for(int j = 0; j < dim; j++) {
        result[(((i / dim) * dim) + j)] += left[i] * right[(((i % dim) * dim) + j)];
        }
    }
}

int main() {
    const int dim = 2048;
    float *left =  new float[dim*dim];
    float *right =  new float[dim*dim];
    float *resultA =  new float[dim*dim];
    float *resultB =  new float[dim*dim];
    float *resultC =  new float[dim*dim];
    for(int i = 0; i < dim*dim; i++) {
        left[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        right[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    clock_t tStart;
    matmulImplNaive(left,right,resultA,dim);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    resultA =  new float[dim*dim];
    tStart = clock();
    matmulFaster(left,right,resultA,dim);
    printf("Time taken (reorder): %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    resultC =  new float[dim*dim];
    double startTime = omp_get_wtime();
    matmulTiling(left,right,resultC,dim);
    printf("Time taken (reorder + tiling): %.2fs\n", (double)(omp_get_wtime() - startTime));
    for(int i = 0; i < dim*dim; i++) {
        if(resultA[i] != resultC[i]) {
            printf("ffs %d",i);
            return 0;
        }
    }

    resultC =  new float[dim*dim];
    startTime = omp_get_wtime();
    matmulTilingMulti(left,right,resultC,dim);
    printf("Time taken (reorder + tiling + multi): %.2fs\n", (double)(omp_get_wtime() - startTime));
    for(int i = 0; i < dim*dim; i++) {
        if(resultA[i] != resultC[i]) {
            printf("ffs %d",i);
            return 0;
        }
    }
    /*
    resultC =  new float[dim*dim];
    startTime = omp_get_wtime();
    matmulNew(left,right,resultC,dim);
    printf("Time taken NEW: %.2fs\n", (double)(omp_get_wtime() - startTime));
    for(int i = 0; i < dim*dim; i++) {
        if(resultA[i] != resultC[i]) {
            printf("ffs %d",i);
            return 0;
        }
    }
    */
    return 0;
}