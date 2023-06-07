#include <iostream>
#include <random>
#include <time.h>

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
int rows = dim;
int columns = dim;
int inners = dim;
  for (int row = 0; row < rows; row++) {
    for (int inner = 0; inner < inners; inner++) {
      for (int col = 0; col < columns; col++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
} } } }

inline void matmulTiling(const float *left, const float *right,
                            float *result, int dim, int tileSize) {
int rows = dim;
int columns = dim;
int inners = dim;

  for(int rowTile = 0; rowTile < rows; rowTile+=tileSize) {
    for (int row = rowTile; row < rowTile+tileSize; row++) {
      for(int innerTile = 0; innerTile < inners; innerTile+=tileSize) {
        for (int inner = innerTile; inner < innerTile+tileSize; inner++) {
          for(int colTile = 0; colTile < columns; colTile+=tileSize) {
            for (int col = colTile; col < colTile+tileSize; col++) {
              result[row * columns + col] +=
                  left[row * columns + inner] * right[inner * columns + col];
            }
          }
        }
      }
    }
  }
}

int main() {
    const int dim = 512;
    float left[dim*dim] = {};
    float right[dim*dim] = {};
    float resultA[dim*dim] = {};
    float resultB[dim*dim] = {};
    float resultC[dim*dim] = {};
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

    int tileSize = 1;
    while(tileSize < dim) {
      float resultC[dim*dim] = {};
      tStart = clock();
      matmulTiling(left,right,resultC,dim,tileSize);
      printf("Time taken (reorder + tiling): %.2fs tileSize = %d \n", (double)(clock() - tStart)/CLOCKS_PER_SEC, tileSize);
      for(int i = 0; i < dim*dim; i++) {
        if(resultA[i] != resultC[i]) {
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