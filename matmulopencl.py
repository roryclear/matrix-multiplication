import numpy as np
import pyopencl as cl

ys = 2;
xs = 4;
platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
local_0 = 256

def matmul(a,b):
	dim = len(a)
	a = a.flatten()
	b = b.flatten()
	a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
	b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

	res_np = np.empty([dim, dim]).astype(np.float32).flatten()
	res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * dim * 4))

	prg = cl.Program(ctx, f"""
	__kernel void matmul(
	    __global const float *a, __global const float *b, __global float *res)
	{{
	int lidx0 = get_local_id(0);
	int startY = lidx0 * ({dim} / {local_0});
	int endY = startY + ({dim} / {local_0});
	for(int y = startY; y < endY; y+={ys}) {{
		for(int x = 0; x < {dim}; x+={xs}) {{
			float acc[2][4];
			acc[0][0] = 0;
			acc[0][1] = 0;
			acc[0][2] = 0;
			acc[0][3] = 0;
			acc[1][0] = 0;
			acc[1][1] = 0;
			acc[1][2] = 0;
			acc[1][3] = 0;
			for(int k = 0; k < {dim}; k++) {{
				acc[0][0] += a[y * {dim} + k] * b[x * {dim} + k];
				acc[0][1] += a[y * {dim} + k] * b[(x + 1) * {dim} + k];
				acc[0][2] += a[y * {dim} + k] * b[(x + 2) * {dim} + k];
				acc[0][3] += a[y * {dim} + k] * b[(x + 3) * {dim} + k];
				acc[1][0] += a[(y + 1) * {dim} + k] * b[(x) * {dim} + k];
				acc[1][1] += a[(y + 1) * {dim} + k] * b[(x + 1) * {dim} + k];
				acc[1][2] += a[(y + 1) * {dim} + k] * b[(x + 2) * {dim} + k];
				acc[1][3] += a[(y + 1) * {dim} + k] * b[(x + 3) * {dim} + k];
			}}
			res[y * {dim} + x] = acc[0][0];
			res[y * {dim} + x + 1] = acc[0][1];
			res[y * {dim} + x + 2] = acc[0][2];
			res[y * {dim} + x + 3] = acc[0][3];
			res[(y + 1) * {dim} + x] = acc[1][0];
			res[(y + 1) * {dim} + x + 1] = acc[1][1];
			res[(y + 1) * {dim} + x + 2] = acc[1][2];
			res[(y + 1) * {dim} + x + 3] = acc[1][3];
		}}
	}}
	}}
	""").build()
	
	knl = prg.matmul
	knl(queue, (local_0,1), (local_0,1), a_g, b_g, res_g) #todo check shape
	cl.enqueue_copy(queue, res_np, res_g)
	return res_np