import numpy as np
import pyopencl as cl

def matmul(a,b,length):
	platform = cl.get_platforms()
	my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
	ctx = cl.Context(devices=my_gpu_devices)
	queue = cl.CommandQueue(ctx)

	mf = cl.mem_flags
	a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
	b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
	res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)


	prg = cl.Program(ctx, """
	__kernel void matmul(
	    __global const float *a, __global const float *b, int length, __global float *res)
	{
	  int gid = get_global_id(0);
	  int row = gid / length;
	  int col = gid % length;
	  float total = 0;
	  if(row < length && col < length) 
	  {
	    for(int i = 0; i < length; i++)
	    {
	      total += a[row * length + i] * b[col + i * length];
	    }
	    res[row * length + col] = total;
	  }
	}
	""").build()
	
	knl = prg.matmul
	knl(queue, a.shape, None, a_g, b_g, np.int32(length), res_g)
	res_np = np.empty_like(a)#fix later
	cl.enqueue_copy(queue, res_np, res_g)
	return res_np