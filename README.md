# Matrix Multiplications

#### 512x512
| implementation  | M2  | MBP16 6-Core i7  |  XPS13  |
|---|---|---|---|
| naive  |   | 0.24s  |   |
| reorder |   | 0.0178s  |   |
| reorder + tiling  |   | 0.0154s  |   |
| avx + tiling  |   | 0.01129s  |   |
| swizzle  |   | 0.00817s  |   |
| swizzle + avx + tiling  |   | 0.00817s  |   |
| avx + tiling + multi  |   | 0.00451s  |   |
| reorder + tiling + multi  |   | 0.00345s  |   |
| swizzle + multi  |   | 0.00213s  |   |
| swizzle + avx + tiling + multi  |   | 0.00175s  |   |
| **numpy**  |   | 0.00252s  |   |

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
| **numpy**  |   | 0.00802s  |   |

#### 2048x2048
| implementation  | M2  | MBP16 6-Core i7  |  XPS13  |
|---|---|---|---|
| naive  |   | 100.57s  |   |
| reorder |   | 1.75s  |   |
| reorder + tiling  |   | 0.57s  |   |
| avx + tiling  |   | 1.79998s  |   |
| swizzle  |   | 0.44771s  |   |
| swizzle + avx + tiling  |   | 0.38539s  |   |
| avx + tiling + multi  |   | 0.92692s  |   |
| reorder + tiling + multi  |   | 0.13381s  |   |
| swizzle + multi  |   | 0.06972s  |   |
| swizzle + avx + tiling + multi  |   | 0.06047s  |   |
| **numpy**  |   | 0.04559  |   |
