#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 22
#define MAX_NUM_THREADS 1024

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
__global__ void cov_coarsening_s2(float * __restrict__  y, const float * __restrict__  x, const float * __restrict__  k){

    // const int B = 10000;
    const int M = 6;
    const int C = 1;
    const int H = 48;
    const int W = 48;
    const int K = 5;
    const int H_out = 44;
    const int W_out = 44;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b = blockIdx.x;
    int m = blockIdx.y;

    int W_grid = ceil((float)W_out / TILE_WIDTH);
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;

    __shared__ float N[26][26];
    __shared__ float M_[5][5];

    if (tx < 5 && ty < 5)
        M_[ty][tx] = k4d(m, 0, ty, tx);

    if (h < 48 && w < 48)
        N[ty][tx] = x4d(b, 0, h, w);

    __syncthreads();

    float acc = 0.0;
    if (ty < 22 && tx < 22 && h < 44 && w < 44){
        #pragma unroll 5
        for (int p = 0; p < 5; ++p){
            #pragma unroll 5
            for (int q = 0; q < 5; ++q)
                acc += N[ty + p][tx + q]*M_[p][q];
        }
            y4d(b, m, h, w) = acc;
    }
    #undef y4d
    #undef x4d
    #undef k4d
    #undef x_shared2d
    #undef k_shared2d
}

__global__ void cov_coarsening_s3(float * __restrict__  y, const float * __restrict__  x, const float * __restrict__  k){

    #define y4d(i3, i2, i1, i0) y[(i3) * 11616 + (i2) * 1936 + (i1) * 44 + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * 2304 + (i2) * 2304 + (i1) * 48 + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * 25 + (i2) * 25 + (i1) * 5 + i0]

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b = blockIdx.x;
    int m = blockIdx.y;

    int W_grid = ceil((float)44 / TILE_WIDTH);
    int h_idx = (blockIdx.z / W_grid) * TILE_WIDTH;
    int w_idx = (blockIdx.z % W_grid) * TILE_WIDTH;
    int h = h_idx + ty;
    int w = w_idx + tx;

    __shared__ float N[22][22];
    __shared__ float M_[5][5];

    if (tx < 5 && ty < 5)
        M_[ty][tx] = k4d(m, 0, ty, tx);

    if (h < 48 && w < 48)
        N[ty][tx] = x4d(b, 0, h, w);
    

    __syncthreads();

    float acc = 0.0;
    if (h < 44 && w < 44){
        #pragma unroll 5
        for (int p = 0; p < 5; ++p){
            #pragma unroll 5
            for (int q = 0; q < 5; ++q){
                if ((ty + p >= 22) || (tx + q >= 22)){
                    if (ty + p + h_idx < 48 && tx + q + w_idx < 48)
                        acc += x4d(b, 0, ty + p + h_idx, tx + q + w_idx)*M_[p][q];
                }else{
                    acc += N[ty + p][tx + q]*M_[p][q];
                }
            }
        }
            y4d(b, m, h, w) = acc;
    }
    #undef y4d
    #undef x4d
    #undef k4d
    #undef x_shared2d
    #undef k_shared2d
}


// ==========================  Optimization 1 : shared memory convolution ==============================
__global__ void opt1(float * __restrict__  y, const float * __restrict__  x, const float * __restrict__  k){

    const int B = 10000;
    const int M = 6;
    const int C = 1;
    const int H = 48;
    const int W = 48;
    const int K = 5;
    const int H_out = 44;
    const int W_out = 44;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int b = blockIdx.x; // batch
    int m = blockIdx.y; //output feature maps

    int h0 = threadIdx.y;
    int w0 = threadIdx.x;

    int W_grid = ceil((float)W_out / TILE_WIDTH);
    int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    int h = h_base + h0;
    int w = w_base + w0;

    int shmem_dim = TILE_WIDTH + K - 1;
    extern __shared__ float shmem[];
    float *x_shared = &shmem[0];
    float *k_shared = &shmem[shmem_dim * shmem_dim];

    #define x_shared2d(i1, i0) x_shared[(i1) * shmem_dim + (i0)]
    #define k_shared2d(i1, i0) k_shared[(i1) * K + (i0)]

    float acc = 0;
    #pragma unroll 15
    for (int c=0; c < 1; ++c){
        if (h0 < K && w0 < K)
            k_shared2d(h0, w0) = k4d(m, c, h0, w0);

        __syncthreads();

        int d = w0 + h0 * TILE_WIDTH;
        int dY = d / shmem_dim;
        int dX = d % shmem_dim;

        int sY = dY + h_base;
        int sX = dX + w_base;

        if (sY >= 0 && sY < H && sX >= 0 && sX < W)
            x_shared2d(dY, dX) = x4d(b, c, sY, sX);
        else
            x_shared2d(dY, dX) = 0.0;


        d = w0 + h0 * TILE_WIDTH + TILE_WIDTH * TILE_WIDTH;
        dY = d / shmem_dim;
        dX = d % shmem_dim;

        sY = dY + h_base;
        sX = dX + w_base;

        if (dY < shmem_dim){
            if (sY >= 0 && sY < H && sX >= 0 && sX < W)
                x_shared2d(dY, dX) = x4d(b, c, sY, sX);
            else
                x_shared2d(dY, dX) = 0.0;
        }
        __syncthreads();

        #pragma unroll 15
        for (int p = 0; p < 5; ++p){
            #pragma unroll 15
            for (int q = 0; q < K; ++q)
                acc += x_shared2d(h0 + p, w0 + q) * k_shared2d(p, q);
        }
                __syncthreads();
    }

        if (h < H_out && w < W_out)
            y4d(b, m, h, w) = acc;

    #undef y4d
    #undef x4d
    #undef k4d
    #undef x_shared2d
    #undef k_shared2d
}


// ==========================  Optimization 2 : constant memory for mask  ==============================
__constant__ float KconstMem[14112];
__global__ void opt2(float * __restrict__ y, const float * __restrict__ x, //const float *k,
		                  const int B, const int M, const int C, const int H, const int W, const int K){

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) KconstMem[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int b = blockIdx.x; // batch
    int m = blockIdx.y; //output feature maps

    int h0 = threadIdx.y;
    int w0 = threadIdx.x;

    int W_grid = ceil((float)W_out / TILE_WIDTH);
    int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    int h = h_base + h0;
    int w = w_base + w0;

    int shmem_dim = TILE_WIDTH + K - 1;
    extern __shared__ float shmem[];
    float *x_shared = &shmem[0];

    #define x_shared2d(i1, i0) x_shared[(i1) * shmem_dim + (i0)]

    float acc = 0;
    for (int c=0; c < C; ++c){
        int d = w0 + h0 * TILE_WIDTH;
	int dY = d / shmem_dim;
	int dX = d % shmem_dim;

	int sY = dY + h_base;
	int sX = dX + w_base;

	if (sY >= 0 && sY < H && sX >= 0 && sX < W)
	    x_shared2d(dY, dX) = x4d(b, c, sY, sX);
	else
	    x_shared2d(dY, dX) = 0.0;



  d = w0 + h0 * TILE_WIDTH + TILE_WIDTH * TILE_WIDTH;
	dY = d / shmem_dim;
	dX = d % shmem_dim;

	sY = dY + h_base;
	sX = dX + w_base;

	if (dY < shmem_dim){
	    if (sY >= 0 && sY < H && sX >= 0 && sX < W)
	        x_shared2d(dY, dX) = x4d(b, c, sY, sX);
	    else
	        x_shared2d(dY, dX) = 0.0;
	}
	__syncthreads();

    #pragma unroll
    for (int p = 0; p < K; ++p)
        #pragma unroll
	    for (int q = 0; q < K; ++q)
	        acc += x_shared2d(h0 + p, w0 + q) * k4d(m,c,p,q);;
	__syncthreads();

    }

    if (h < H_out && w < W_out)
        y4d(b, m, h, w) = acc;

    #undef y4d
    #undef x4d
    #undef k4d
    #undef x_shared2d
}

// Layer 2 -------- //
__global__ void layer_2_16x32_half(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

    #define x4d(i3, i2, i1, i0) x[(i3) * 2904 + (i2) * 484 + (i1) * 22 + i0]
    
        int ty = threadIdx.y;
        int tx = threadIdx.x;
        int Row_base = blockIdx.y * (8 * blockDim.y) + threadIdx.y;
        int Col_base = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
        // use z (batch No.) to offset indexing when loading X & writing Y
        int b = blockIdx.z;
    
        // rc for register cache
        half2 rc_W[8];
        half2 rc_X[8];
        half2 rc_result[8] = { __float2half2_rn(0) }; // zero out
    
        int m, Row, Col, Row_x, c, kk, i, t, n;
        unsigned mask = 0xffffffff; // __activemask();
        half temp1, temp2;
    
        #pragma unroll
        for (t = 0; t < 10; t++) {
            m = t * 16; // Col in W & Row in X
    
            // Load tileW & tileX into register cache
            // W
            #pragma unroll
            for (i = 0; i < 8; i++) {
                Row = Row_base + i * blockDim.y;
                if (Row < 16 && m + tx < 150)
                    rc_W[i] = __half2half2(__float2half(k[Row*150 + (m + tx)]));
                else
                    rc_W[i] = __float2half2_rn(0);
            }
            // X
            #pragma unroll
            for (i = 0; i < 8; i++) {
                Row_x = m + i * blockDim.y + ty;
                c = Row_x / 25;
                kk = Row_x % 25;
    
                Col = Col_base;
                if (Row_x < 150 && Col < 324)
                    temp1 = __float2half(x4d(b, c, Col/18 + kk/5, Col%18 + kk%5));
                else
                    temp1 = __float2half(0);
    
                Col = Col_base + blockDim.x;
                if (Row_x < 150 && Col < 324)
                    temp2 = __float2half(x4d(b, c, Col/18 + kk/5, Col%18 + kk%5));
                else
                    temp2 = __float2half(0);
    
                rc_X[i] = __halves2half2(temp1, temp2);
            }
    
            // register cache, no sync needed
    
            // Compute & Accumulate result of each tile (with warp-level shared)
            // int tid, val;
            #pragma unroll
            for (i = 0; i < 8; i++) {
                #pragma unroll
                for (n = 0; n < 16; n++) {
                    // tid = n % 2; // thread ty in the warp to be queried
                    // val = (n / 2) * 2 + j; // register value id of the query thread
                    rc_result[i] = __hfma2(__shfl_sync(mask, rc_W[i], ty * 16 + n), __shfl_sync(mask, rc_X[n/2], n%2 * 16 + tx), rc_result[i]);
                }
            }
    
        }
    
        // Write
        #pragma unroll
        for (i = 0; i < 8; i++) {
            Row = Row_base + i * blockDim.y;
            if (Row < 16 && Col_base < 324)
                y[b * 5184 + Row * 324 + Col_base] = __low2float(rc_result[i]);
            if (Row < 16 && Col_base + blockDim.x < 324)
                y[b * 5184 + Row * 324 + Col_base + blockDim.x] = __high2float(rc_result[i]);
        }
    
    #undef x4d
    }

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0]; // batch
    const int M = y.shape_[1]; // num of features in output
    const int C = x.shape_[1]; // num of features in input
    const int H = x.shape_[2]; // height of input image
    const int W = x.shape_[3]; // width of input image
    const int K=  w.shape_[3]; // dim of convolutional kernel



    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // printf("B: %d; M: %d; C: %d; H: %d; W: %d; K: %d\n",B,M,C,H,W,K);
    printf("x shape:%d, %d, %d, %d\n",x.shape_[0],x.shape_[1],x.shape_[2], x.shape_[3]);
    printf("y shape:%d, %d, %d, %d\n",y.shape_[0],y.shape_[1],y.shape_[2], y.shape_[3]);
    printf("w shape:%d, %d, %d, %d\n",w.shape_[0],w.shape_[1],w.shape_[2], w.shape_[3]);
    int W_grid = ceil((float)W_out / TILE_WIDTH);
    int H_grid = ceil((float)H_out / TILE_WIDTH);
    int Z = H_grid * W_grid;

    // optimization 1

    // int shmem_dim = TILE_WIDTH + K - 1;
    // size_t shmem_size = sizeof(float) * (shmem_dim * shmem_dim + K * K);
    // printf("shared mem convolution\n");
    // opt1<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    
    //end of optimization 1

    // optimization 2

    
    if (x.shape_[2] == 48) {
        // cudaMemcpyToSymbol(KconstMem, w.dptr_, M * C * K * K * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        // size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1));
        // opt2<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_, B,M,C,H,W,K);
        // dim3 blockDim(TILE_WIDTH + 4,TILE_WIDTH + 4,1);
        dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
        dim3 gridDim(B,M,Z);
        cov_coarsening_s3<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_);
        // opt_new<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,w.dptr_);
    } else {
        dim3 gridDim(ceil(324.0/32), ceil(16.0/16), B);
        dim3 blockDim(16, 2, 1);
        layer_2_16x32_half<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);
    }
    //end of optimization 2

    // end end of optimization 6
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}




/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
