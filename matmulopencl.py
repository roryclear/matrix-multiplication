import numpy as np
import pyopencl as cl

def matmul(a,b):
	dim = len(a)
	#print("dim =",dim)
	ys = 32;
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

	local_0 = 8
	prg = cl.Program(ctx, f"""
	__kernel void matmul(
	    __global const float *a, __global const float *b, __global float *res)
	{{
	int lidx0 = get_local_id(0);
	int startY = lidx0 * ({dim} / {local_0});
	int endY = startY + ({dim} / {local_0});
	for(int y = startY; y < endY; y+={ys}) {{
		for(int x = 0; x < {dim}; x+={xs}) {{
			for(int iy = y; iy < (y+{ys}); iy++) {{
				float total = 0;
				float total2 = 0;
				for(int k = 0; k < {dim}; k++) {{
					total += a[iy * {dim} + k] * b[x * {dim} + k];
					total2 += a[iy * {dim} + k] * b[(x + 1) * {dim} + k];
				}}
				res[iy * {dim} + x] = total;
				res[iy * {dim} + x + 1] = total2;
			}}
		}}
	}}
	}}
	""").build()
	
	knl = prg.matmul
	knl(queue, (local_0,1), (local_0,1), a_g, b_g, res_g) #todo check shape
	cl.enqueue_copy(queue, res_np, res_g)
	return res_np