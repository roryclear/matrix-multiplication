import numpy as np
import matmulmetal2 as mmm
import matmulopencl as mmo
import time

length = 8

a = np.random.rand(length,length).astype(np.float32)
b = np.random.rand(length,length).astype(np.float32)

print("a =",a)
print("\nb =",b)
#a = np.empty_like(a)
#b = np.empty_like(b)
answer = np.empty_like(a)

b2 = np.zeros_like(b)

#for y in range(length):
#	for x in range(length):
#		b2[y][x] = b[x][y];

start_time = time.time()
om = mmm.matmul(a,b)
print("--- metal %.5f seconds ---" % (time.time() - start_time))


start_time = time.time()
answer = np.matmul(a,b).flatten()
print("--- numpy %.5f seconds ---" % (time.time() - start_time))

assert np.allclose(om, answer)
print("passed")