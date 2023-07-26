# Matrix Multiplications
#### 1024x1024
| implementation  | M2  | MBP16 6-Core i7  |  XPS13  |
|---|---|---|---|
| naive  |   | 2.16s  |   |
| reorder |   | 0.11s  |   |
| reorder + tiling  |   | 0.08s  |   |
| avx + tiling  |   | 0.06631s  |   |
| swizzle  |   | 0.04215s  |   |
| swizzle + avx + tiling  |   | 0.03816s  |   |
| avx + tiling + multi  |   | 0.02268s  |   |
| reorder + tiling + multi  |   | 0.02070s  |   |
| swizzle + multi  |   | 0.00914s  |   |
| swizzle + avx + tiling + multi  |   | 0.00704s  |   |
