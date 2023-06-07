#include <iostream>
#include <random>
#include <time.h>

//g++ -O3 -march=native -ffast-math matmul.cpp -o a

inline void matmulImplNaive(const float *left, const float *right,
                            float *result, int dim) {
int rows = dim;
int columns = dim;
int inners = dim;
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      for (int inner = 0; inner < inners; inner++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
} } } }


inline void matmulFaster(const float *left, const float *right,
                            float *result, int dim) {
  for (int row = 0; row < dim; row++) {
    for (int inner = 0; inner < dim; inner++) {
      for (int col = 0; col < dim; col++) {
        result[row * dim + col] +=
            left[row * dim + inner] * right[inner * dim + col];
} } } }


inline void matmulTiling(const float *left, const float *right,
                            float *result, int dim, int tileSize) {
  for(int rowTile = 0; rowTile < dim; rowTile+=tileSize) {
    for (int innerTile = 0; innerTile < dim; innerTile+=tileSize) {
      for(int colTile = 0; colTile < dim; colTile+=tileSize) {
        for (int row = rowTile; row < rowTile+tileSize; row++) {
          for(int inner = innerTile; inner < innerTile+tileSize; inner++) {
            for (int col = colTile; col < colTile+tileSize; col++) {
              result[row * dim + col] +=
                  left[row * dim + inner] * right[inner * dim + col];
            }
          }
        }
      }
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
    clock_t tStart = clock();
    matmulImplNaive(left,right,resultA,dim);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    tStart = clock();
    matmulFaster(left,right,resultB,dim);
    printf("Time taken (reorder): %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    int tileSize = 2;
    while(tileSize < dim) {
      resultC =  new float[dim*dim];
      tStart = clock();
      matmulTiling(left,right,resultC,dim,tileSize);
      printf("Time taken (reorder + tiling): %.2fs tileSize = %d \n", (double)(clock() - tStart)/CLOCKS_PER_SEC, tileSize);
      for(int i = 0; i < dim*dim; i++) {
          if(resultA[i] != resultC[i]) {
              printf("ffs %d",i);
              return 0;
          }
      }
      tileSize += tileSize;
    }

    for(int i = 0; i < dim*dim; i++) {
        if(resultA[i] != resultB[i]) {
            return 0;
        }
    }
    printf("reorder outputs equal\n");

    for(int i = 0; i < dim*dim; i++) {
        if(resultA[i] != resultC[i]) {
            return 0;
        }
    }

    printf("tiling outputs equal");

    return 0;
}