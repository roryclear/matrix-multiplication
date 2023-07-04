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
// alias g++=/usr/local/bin/g++-13

#define by 4
#define bx 2

const int dim = 1024;

float *left =  new float[dim*dim];
float *right =  new float[dim*dim];
float *rightr =  new float[dim*dim];
float *rightr2 =  new float[dim*dim];
float *resultA =  new float[dim*dim];
float *resultB =  new float[dim*dim];
float *resultC =  new float[dim*dim];

inline void matmulImplNaive() {
  for (int y = 0; y < dim; y++) {
    for (int x = 0; x < dim; x++) {
      for (int k = 0; k < dim; k++) {
        resultA[y * dim + x] +=
            left[y * dim + k] * right[k * dim + x];
      } 
    } 
  } 
}


inline void matmulFaster() {
  for (int y = 0; y < dim; y++) {
    for (int k = 0; k < dim; k++) {
      float lnum = left[y * dim + k];
      for (int x = 0; x < dim; x++) {
        resultA[y * dim + x] +=
            lnum * right[k * dim + x];
      } 
    } 
  } 
}

inline void matmulSwizzle() {
  for (int y = 0; y < dim; y+=by) {
    for (int x = 0; x < dim; x+=bx) {
      float acc[by][bx] = {};
      for (int k = 0; k < dim; k++) {
        for(int iy = 0; iy < by; iy++) {
          float lnum = left[(y+iy)*dim + k];
          for(int ix = 0; ix < bx; ix++) {
              acc[iy][ix] += lnum * rightr[(x+ix)*dim + k];
          }
        }
      }

      for(int iy = 0; iy < by; iy++) {
        for(int ix = 0; ix < bx; ix++) {
          resultC[(y+iy)*dim + x + ix] = acc[iy][ix];
        }
      }

    } 
  } 
}

inline void matmulSwizzleAvx() {
  __m256 *rightm = (__m256*)rightr2;
  __m256 *resultm = (__m256*)resultC;
  for (int y = 0; y < dim; y+=by) {
    for (int x = 0; x < dim; x+=bx*8) {
      __m256 accm[by][bx] = {};
      for (int k = 0; k < dim; k++) {
        for(int iy = 0; iy < by; iy++) {
          __m256 am = _mm256_broadcast_ss(&left[(y+iy)*dim + k]);
          for(int ix = 0; ix < bx; ix++) {
            accm[iy][ix] = _mm256_fmadd_ps(am,rightm[((x + ix * 8)*dim + k*8)/8],accm[iy][ix]);
          }
        }
      }
      for(int iy = 0; iy < by; iy++) {
        for(int ix = 0; ix < bx; ix++) {
          resultm[((y+iy)*dim + x + ix * 8)/8] = accm[iy][ix];
        }
      }
    } 
  } 
}

inline void matmulAvx2() {
  __m256 *rightm = (__m256*)right;
  __m256 *resultm = (__m256*)resultC;
  #pragma omp parallel for
  for(int y = 0; y < dim; y+=by) {
    for(int x = 0; x < dim; x+=bx*8) {
      __m256 accm[by][bx] = {};
      for(int k = 0; k < dim; k++) {
        for(int iy = 0; iy < by; iy++) {
            __m256 ta = _mm256_broadcast_ss(&left[((y+iy)*dim) + k]);
            for(int ix = 0; ix < bx*8; ix+=8) {
              accm[iy][ix/8] = _mm256_fmadd_ps(ta, rightm[((k*dim) + x + ix)/8],accm[iy][ix/8]);
               //resultm[((y+iy)*dim + x + ix)/8] = _mm256_fmadd_ps(ta, rightm[((k*dim) + x + ix)/8], resultm[((y+iy)*dim + x + ix)/8]);
            }
         }
      }
      for(int iy = 0; iy < by; iy++) {
        for(int ix = 0; ix < bx*8; ix+=8) {
          resultm[((y+iy)*dim + x + ix)/8] = accm[iy][ix/8];
        }
      }
    }
  }
}

inline void matmulSwizzleAvxMulti() {
  __m256 *rightm = (__m256*)rightr2;
  __m256 *resultm = (__m256*)resultC;
  #pragma omp parallel for
  for (int y = 0; y < dim; y+=by) {
    for (int x = 0; x < dim; x+=bx*8) {
      __m256 accm[by][bx] = {};
      for (int k = 0; k < dim; k++) {
        for(int iy = 0; iy < by; iy++) {
          __m256 am = _mm256_broadcast_ss(&left[(y+iy)*dim + k]);
          for(int ix = 0; ix < bx; ix++) {
            accm[iy][ix] = _mm256_fmadd_ps(am,rightm[((x + ix * 8)*dim + k*8)/8],accm[iy][ix]);
          }
        }
      }
      for(int iy = 0; iy < by; iy++) {
        for(int ix = 0; ix < bx; ix++) {
          resultm[((y+iy)*dim + x + ix * 8)/8] = accm[iy][ix];
        }
      }
    } 
  } 
}

inline void matmulSwizzleMulti() {
  #pragma omp parallel for
  for (int y = 0; y < dim; y+=by) {
    for (int x = 0; x < dim; x+=bx) {
      float acc[by][bx] = {};
      for (int k = 0; k < dim; k++) {
        for(int iy = 0; iy < by; iy++) {
          float lnum = left[(y+iy)*dim + k];
          for(int ix = 0; ix < bx; ix++) 
          {
            acc[iy][ix] += lnum * rightr[(x+ix)*dim + k];
          }
        }
      }

      for(int iy = 0; iy < by; iy++) {
        for(int ix = 0; ix < bx; ix++) {
          resultC[(y+iy)*dim + x + ix] = acc[iy][ix];
        }
      }

    } 
  } 
}


inline void matmulTiling() {
  int xs = 256;
  int ys = 16;
  int ks = 8;
  for(int yt = 0; yt < dim; yt+=ys) {
    for (int kt = 0; kt < dim; kt+=ks) {
      for(int xt = 0; xt < dim; xt+=xs) {
        for (int y = yt; y < yt+ys; y++) {
          for(int k = kt; k < kt+ks; k++) {
            float lnum = left[y * dim + k];
            for (int x = xt; x < xt+xs; x++) {
              resultC[y * dim + x] +=
                  lnum * right[k * dim + x];
            }
          }
        }
      }
    }
  }
}


//1024 4 4 best on xps
// 256 16 8 best on macbook
inline void matmulTilingMulti() {
  int xs = 256;
  int ys = 16;
  int ks = 4;
  #pragma omp parallel for
  for(int yt = 0; yt < dim; yt+=ys) {
    for (int kt = 0; kt < dim; kt+=ks) {
      for(int xt = 0; xt < dim; xt+=xs) {
        for (int y = yt; y < yt+ys; y++) {
          for(int k = kt; k < kt+ks; k++) {
            float lnum = left[y * dim + k];
            for (int x = xt; x < xt+xs; x++) {
              resultC[y * dim + x] +=
                  lnum * right[k * dim + x];
            }
          }
        }
      }
    }
  }
}



int main() {
  for(int i = 0; i < dim*dim; i++) {
      left[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      right[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }

  clock_t tStart;
  matmulImplNaive();
  printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

  resultA =  new float[dim*dim];
  tStart = clock();
  matmulFaster();
  printf("Time taken (reorder): %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

  resultC =  new float[dim*dim];
  double startTime = omp_get_wtime();
  matmulTiling();
  printf("Time taken (reorder + tiling): %.2fs\n", (double)(omp_get_wtime() - startTime));
  for(int i = 0; i < dim*dim; i++) {
      if(resultA[i] != resultC[i]) {
          printf("ffs %d",i);
          return 0;
      }
  }

  for(int y = 0; y < dim; y++) {
    for(int x = 0; x < dim; x++) {
      rightr[y * dim + x] = right[x * dim + y];
    }
  }


  resultC =  new float[dim*dim];
  startTime = omp_get_wtime();
  matmulSwizzle();
  printf("Time taken (swizzle): %.5fs\n", (double)(omp_get_wtime() - startTime));
  for(int i = 0; i < dim*dim; i++) {
      if(abs(resultC[i] - resultA[i]) > abs(resultA[i]*0.00001)) {
          float diff = abs(resultC[i] - resultA[i]);
          printf("ffs %d %f -> %f %f\n",i,resultC[i],resultA[i],diff);
          return 0;
      }
  }

  resultC =  new float[dim*dim];
  startTime = omp_get_wtime();
  matmulTilingMulti();
  printf("Time taken (reorder + tiling + multi): %.2fs\n", (double)(omp_get_wtime() - startTime));
  for(int i = 0; i < dim*dim; i++) {
      if(resultA[i] != resultC[i]) {
          printf("ffs %d",i);
          return 0;
      }
  }

  resultC =  new float[dim*dim];
  startTime = omp_get_wtime();
  matmulSwizzleMulti();
  printf("Time taken (swizzle + multi): %.5fs\n", (double)(omp_get_wtime() - startTime));
  for(int i = 0; i < dim*dim; i++) {
      if(abs(resultC[i] - resultA[i]) > abs(resultA[i]*0.00001)) {
          float diff = abs(resultC[i] - resultA[i]);
          printf("ffs %d %f -> %f %f\n",i,resultC[i],resultA[i],diff);
          return 0;
      }
  }

  for(int y = 0; y < dim; y+=8) {
    for(int x = 0; x < dim; x++) {
      for(int iy = 0; iy < 8; iy++) {
        //rightr[y*dim + x*8 +iy] = right[(y+iy)*dim + x];
        rightr2[y*dim + x*8 + iy] = rightr[(y+iy)*dim + x]; //each 256 contains y,x -> y+7,x
        //right2[(x + ix)*dim + k + i] = right[(x+ix + i)*dim + k]
      }
    }
  }

  
  resultC =  new float[dim*dim];
  startTime = omp_get_wtime();
  matmulSwizzleAvx();
  printf("Time taken (swizzle + avx): %.5fs\n", (double)(omp_get_wtime() - startTime));
  for(int i = 0; i < dim*dim; i++) {
      if(abs(resultC[i] - resultA[i]) > abs(resultA[i]*0.00001)) {
          float diff = abs(resultC[i] - resultA[i]);
          printf("ffs %d %f -> %f %f\n",i,resultC[i],resultA[i],diff,rightr2);
          return 0;
      }
  }

  resultC =  new float[dim*dim];
  startTime = omp_get_wtime();
  matmulSwizzleAvxMulti();
  printf("Time taken (swizzle + avx + multi): %.5fs\n", (double)(omp_get_wtime() - startTime));
  for(int i = 0; i < dim*dim; i++) {
      if(abs(resultC[i] - resultA[i]) > abs(resultA[i]*0.00001)) {
          float diff = abs(resultC[i] - resultA[i]);
          printf("ffs %d %f -> %f %f\n",i,resultC[i],resultA[i],diff,rightr2);
          return 0;
      }
  }

  resultC =  new float[dim*dim];
  startTime = omp_get_wtime();
  matmulAvx2();
  printf("Time taken (AVX2): %.5fs\n", (double)(omp_get_wtime() - startTime));
  for(int i = 0; i < dim*dim; i++) {
      if(abs(resultC[i] - resultA[i]) > abs(resultA[i]*0.00001)) {
          float diff = abs(resultC[i] - resultA[i]);
          printf("ffs %d %f -> %f %f\n",i,resultC[i],resultA[i],diff,rightr2);
          return 0;
      }
  }

  return 0;
}