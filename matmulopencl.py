import numpy as np
import pyopencl as cl

def matmul(a,b,length):
	aRows = a.shape[0]
	aCols = a.shape[1]
	bRows = b.shape[0]
	bCols = b.shape[1]
	a = a.flatten()
	b = b.flatten()
	platform = cl.get_platforms()
	my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
	ctx = cl.Context(devices=my_gpu_devices)
	queue = cl.CommandQueue(ctx)

	mf = cl.mem_flags
	a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
	b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

	res_np = np.empty([aRows, bCols]).astype(np.float32).flatten()
	res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (aRows * bCols * 4))

	prg = cl.Program(ctx, """
	__kernel void matmul(
	    __global const float *a, __global const float *b, int aRows, int aCols, int bCols, __global float *res)
	{
	  int gid = get_global_id(0);
	  int row = gid / bCols;
	  int col = gid % bCols;
	  float total = 0;
	  if(row < aRows && col < bCols) 
	  {
	    for(int i = 0; i < aCols; i++)
	    {
	      total += a[row * aCols + i] * b[col + i * bCols];
	    }
	    res[row * bCols + col] = total;
	  }
	}
	""").build()
	
	knl = prg.matmul
	knl(queue, (aRows*bCols,1), None, a_g, b_g, np.int32(aRows), np.int32(aCols), np.int32(bCols), res_g) #todo check shape
	cl.enqueue_copy(queue, res_np, res_g)
	return res_np