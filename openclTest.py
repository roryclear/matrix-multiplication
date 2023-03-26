import numpy as np
import matmulopencl as mmo

length = 128

a = np.random.rand(200,length).astype(np.float32)
b = np.random.rand(length,190).astype(np.float32)
answer = np.matmul(a,b).flatten()
output = mmo.matmul(a,b)
assert np.allclose(output, answer)
print("passed")