import numpy as np
import matmulmetal as mmm
import matmulopencl as mmo
import time

length = 256
size = length*length

a = np.random.rand(size).astype(np.float32)
b = np.random.rand(size).astype(np.float32)
answer = np.empty_like(a)
om = np.empty_like(a)
oo = np.empty_like(a)

start_time = time.time()
om = mmm.matmul(a,b,length)
print("--- metal %s seconds ---" % (time.time() - start_time))

start_time = time.time()
oo = mmo.matmul(a,b,length)
print("--- opencl %s seconds ---" % (time.time() - start_time))


for r in range(length):
  for c in range(length):
    total = 0
    for n in range(length):
      total += a[r * length + n] * b[c + n * length]
    answer[r * length + c] = total

assert np.allclose(om, answer)
assert np.allclose(oo, answer)
print("passed")