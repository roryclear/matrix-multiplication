import numpy as np
import matmulopencl as mmo
import time
import torch
#import tensorflow as tf

length = 1024

a = np.random.rand(length,length).astype(np.float32)
b = np.random.rand(length,length).astype(np.float32)
#a = np.ones((length,length),dtype=np.float32)
#b = np.ones((length,length),dtype=np.float32)

answer = np.empty_like(a)

opencl_time = None
for _ in range(20):
	start_time = time.perf_counter()
	oo = mmo.matmul(a,b)
	t = time.perf_counter() - start_time
	if opencl_time == None or t < opencl_time:
		opencl_time = t
		
print("--- opencl\t%.5f seconds ---" % (opencl_time))

fastest = None
for _ in range(20):
	start_time = time.perf_counter()
	answer = np.matmul(a,b).flatten()
	t = time.perf_counter() - start_time
	if fastest == None or t < fastest:
		fastest = t
print("--- numpy\t%.5f seconds ---" % (fastest))


at = torch.from_numpy(a)
bt = torch.from_numpy(b)
fastest = None
for _ in range(20):
	start_time = time.perf_counter()
	x = torch.matmul(at,bt).flatten()
	t = time.perf_counter() - start_time
	if fastest == None or t < fastest:
		fastest = t
print("--- torch\t%.5f seconds ---" % (fastest))

assert np.allclose(oo, answer)
print("passed")