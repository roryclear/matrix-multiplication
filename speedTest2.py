import numpy as np
import matmulmetal2 as mmm
import matmulopencl as mmo
import time

length = 2048

a = np.random.rand(length,length).astype(np.float32)
b = np.random.rand(length,length).astype(np.float32)
answer = np.empty_like(a)

start_time = time.time()
om = mmm.matmul(a,b)
print("--- metal %.5f seconds ---" % (time.time() - start_time))


start_time = time.time()
answer = np.matmul(a,b).flatten()
print("--- numpy %.5f seconds ---" % (time.time() - start_time))

assert np.allclose(om, answer)
print("passed")