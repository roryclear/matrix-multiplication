# Matrix Multiplications

#### 512x512
| implementation  | M2  | MBP16 6-Core i7  |  XPS13  |
|---|---|---|---|
| naive  |   | 0.24s  | 0.23s |
| reorder |   | 0.0178s  | 0.0125s |
| reorder + tiling  |   | 0.0154s  | 0.0125s |
| avx + tiling  |   | 0.01129s  | 0.01090s |
| swizzle  |   | 0.00817s  | 0.00760s |
| swizzle + avx + tiling  |   | 0.00817s  | 0.00640s |
| avx + tiling + multi  |   | 0.00451s  | 0.00530s |
| reorder + tiling + multi  |   | 0.00345s  | broken |
| swizzle + multi  |   | 0.00213s  | 0.00220s |
| swizzle + avx + tiling + multi  |   | 0.00175s  | 0.00200s |
| **numpy**  |   | **0.00252s** |  |

#### 1024x1024
| implementation  | M2  | MBP16 6-Core i7  |  XPS13  |
|---|---|---|---|
| naive  |   | 2.16s  | 2.29s |
| reorder |   | 0.11s  | 0.0900s |
| reorder + tiling  |   | 0.08s  | 0.0700s |
| avx + tiling  |   | 0.06631s  | 0.04710s |
| swizzle  |   | 0.04215s  | 0.03040s |
| swizzle + avx + tiling  |   | 0.03816s  | 0.02590s |
| avx + tiling + multi  |   | 0.02268s  | 0.02100s |
| reorder + tiling + multi  |   | 0.02070s  | 0.02330s |
| swizzle + multi  |   | 0.00914s  | 0.01000s |
| swizzle + avx + tiling + multi  |   | 0.00704s  | 0.00940s |
| **numpy**  |   | **0.00802s** | **0.01557s** |

#### 2048x2048
| implementation  | M2  | MBP16 6-Core i7  |  XPS13  |
|---|---|---|---|
| naive  |   | 100.57s  | 61.56s |
| reorder |   | 1.75s  | 1.4176s |
| reorder + tiling  |   | 0.57s  | 0.8197s |
| avx + tiling  |   | 1.79998s  | 1.51850s |
| swizzle  |   | 0.44771s  | 0.40390s |
| swizzle + avx + tiling  |   | 0.38539s  | 0.39300s |
| avx + tiling + multi  |   | 0.92692s  | 0.55380s |
| reorder + tiling + multi  |   | 0.13381s  | 0.19100s |
| swizzle + multi  |   | 0.06972s  | 0.09370s |
| swizzle + avx + tiling + multi  |   | 0.06047s  | 0.08410s |
| **numpy**  |   | **0.04559** | **0.06244** |
