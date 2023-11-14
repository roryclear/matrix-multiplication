import numpy as np
import sum_metal as sm
import matmulopencl as mmo
import time
import torch
#import tensorflow as tf

length = 4096*4

a = np.random.rand(length).astype(np.float32)
fastest = None
answer = None
for i in range(20):
	start_time = time.time()
	answer = np.sum(a)
	t = time.time() - start_time
	if fastest == None or t < start_time:
		fastest = t
print("--- numpy\t%.5f seconds ---" % (fastest))

metal_time = None
for _ in range(20):
	start_time = time.time()
	om = sm.sum(a)
	t = time.time() - start_time
	if metal_time == None or t < metal_time:
		metal_time = t
print("--- metal\t%.5f seconds ---" % (metal_time))

assert np.allclose(om, answer)
print("passed")