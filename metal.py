#!/usr/bin/env python

import numpy as np
import pyopencl as cl
import time

length = 32
size = length*length

a_np = np.random.rand(size).astype(np.float32)
b_np = np.random.rand(size).astype(np.float32)

answer = np.empty_like(a_np)

start_time = time.time()
for r in range(length):
  for c in range(length):
    total = 0
    for n in range(length):
      total += a_np[r * length + n] * b_np[c + n * length]
    answer[r * length + c] = total
print("--- py %s seconds ---" % (time.time() - start_time))

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
#ctx = cl.create_some_context() for chosing platform? 

queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

prg = cl.Program(ctx, """
__kernel void matmul(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int length = 32;
  int gid = get_global_id(0);
  int row = gid / length;
  int col = gid % length;
  float total = 0;
  if(row < length && col < length) 
  {
    for(int i = 0; i < length; i++)
    {
      total += a_g[row * length + i] * b_g[col + i * length];
    }
    res_g[row * length + col] = total;
  }
}
""").build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
knl = prg.matmul  # Use this Kernel object for repeated calls
knl(queue, a_np.shape, None, a_g, b_g, res_g)

res_np = np.empty_like(a_np)
start_time = time.time()
cl.enqueue_copy(queue, res_np, res_g)
print("--- opencl %s seconds ---" % (time.time() - start_time))

# Check on CPU with Numpy:
print(answer - res_np)
assert np.allclose(res_np, answer)