#include <iostream>
#include <random>

inline void matmulImplNaive(const float *left, const float *right,
                            float *result) {
int rows = 64;
int columns = 64;
int inners = 64;
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      for (int inner = 0; inner < inners; inner++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
} } } }

int main() {
    float left[64*64] = {};
    float right[64*64] = {};
    float result[64*64] = {};
    for(int i = 0; i < 64*64; i++) {
        left[i] = rand();
        right[i] = rand();
    }
    matmulImplNaive(left,right,result);
    for(int i = 0; i < 64*64; i++) {
        std::cout << result[i];
    }
    std::cout << "matmul??";
    return 0;
}