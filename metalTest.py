import numpy as np
import matmulmetal as mmm
import matmulopencl as mmo
import time
import torch
#import tensorflow as tf

length = 2048

a = np.random.rand(length,length).astype(np.float32)
b = np.random.rand(length,length).astype(np.float32)

answer = np.empty_like(a)

b2 = np.zeros_like(b)

metal_time = None
for _ in range(20):
	start_time = time.time()
	om = mmm.matmul(a,b)
	t = time.time() - start_time
	if metal_time == None or t < metal_time:
		metal_time = t
print("--- metal\t%.5f seconds ---" % (metal_time))

fastest = None
for _ in range(20):
	start_time = time.time()
	answer = np.matmul(a,b).flatten()
	t = time.time() - start_time
	if fastest == None or t < fastest:
		fastest = t
print("--- numpy\t%.5f seconds ---" % (fastest))

at = torch.from_numpy(a)
bt = torch.from_numpy(b)
fastest = None
for _ in range(20):
	start_time = time.time()
	x = torch.matmul(at,bt).flatten()
	t = time.time() - start_time
	if fastest == None or t < fastest:
		fastest = t
print("--- torch\t%.5f seconds ---" % (fastest))

assert np.allclose(om, answer)
print("passed")