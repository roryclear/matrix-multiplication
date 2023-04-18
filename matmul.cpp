#include <iostream>
#include <random>

inline void matmulImplNaive(const float *left, const float *right,
                            float *result) {
int dim = 64;
int rows = dim;
int columns = dim;
int inners = dim;
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      for (int inner = 0; inner < inners; inner++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
} } } }

int main() {
    const int dim = 64;
    float left[dim*dim] = {};
    float right[dim*dim] = {};
    float result[dim*dim] = {};
    for(int i = 0; i < dim*dim; i++) {
        left[i] = rand();
        right[i] = rand();
    }
    matmulImplNaive(left,right,result);
    for(int i = 0; i < dim*dim; i++) {
        std::cout << result[i];
    }
    std::cout << "matmul??";
    return 0;
}