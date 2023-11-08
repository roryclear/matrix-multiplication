import numpy as np
import matmulmetal2 as mmm
import matmulopencl as mmo
import time
import torch
import tensorflow as tf

length = 2048

a = np.random.rand(length,length).astype(np.float32)
b = np.random.rand(length,length).astype(np.float32)

answer = np.empty_like(a)

b2 = np.zeros_like(b)

start_time = time.time()
om = mmm.matmul(a,b)
metal_time = time.time() - start_time
print("--- metal\t%.5f seconds ---" % (metal_time))

start_time = time.time()
answer = np.matmul(a,b).flatten()
print("--- numpy\t%.5f seconds ---" % (time.time() - start_time))

at = torch.from_numpy(a)
bt = torch.from_numpy(b)
start_time = time.time()
t = torch.matmul(at,bt).flatten()
print("--- torch\t%.5f seconds ---" % (time.time() - start_time))

start_time = time.time()
c = tf.matmul(a, b)
print("--- tensorflow\t%.5f seconds ---" % (time.time() - start_time))

assert np.allclose(om, answer)
print("passed")