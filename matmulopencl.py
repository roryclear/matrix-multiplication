import numpy as np
import pyopencl as cl

def matmul(a,b):
	dim = len(a)
	#print("dim =",dim)
	ys = 128;
	xs = 2;

	a = a.flatten()
	b = b.flatten()
	platform = cl.get_platforms()
	my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
	ctx = cl.Context(devices=my_gpu_devices)
	queue = cl.CommandQueue(ctx)

	mf = cl.mem_flags
	a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
	b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

	res_np = np.empty([dim, dim]).astype(np.float32).flatten()
	res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * dim * 4))

	prg = cl.Program(ctx, f"""
	__kernel void matmul(
	    __global const float *a, __global const float *b, __global float *res)
	{{
	for(int y = 0; y < {dim}; y+={ys}) {{
		for(int x = 0; x < {dim}; x+={xs}) {{
			for(int iy = y; iy < (y+{ys}); iy++) {{
				for(int ix = x; ix < (x+{xs}); ix++) {{
					float total = 0;
					for(int k = 0; k < {dim}; k++) {{
						total += a[iy * {dim} + k] * b[ix * {dim} + k];
					}}
					res[iy * {dim} + ix] = total;
				}}
			}}
		}}
	}}
	}}
	""").build()
	
	knl = prg.matmul
	knl(queue, (1,1), None, a_g, b_g, res_g) #todo check shape
	cl.enqueue_copy(queue, res_np, res_g)
	return res_np