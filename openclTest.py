import numpy as np
import matmulopencl as mmo

length = 128
size = length*length

a = np.random.rand(size).astype(np.float32)
b = np.random.rand(size).astype(np.float32)
answer = np.empty_like(a)
output = np.empty_like(a)

output = mmo.matmul(a,b,length)

for r in range(length):
  for c in range(length):
    total = 0
    for n in range(length):
      total += a[r * length + n] * b[c + n * length]
    answer[r * length + c] = total

assert np.allclose(output, answer)
print("passed")