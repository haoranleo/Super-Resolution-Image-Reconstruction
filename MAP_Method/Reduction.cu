/**
 *   ___ _   _ ___   _     __  __   _   ___     
 *  / __| | | |   \ /_\   |  \/  | /_\ | _ \    
 * | (__| |_| | |) / _ \  | |\/| |/ _ \|  _/    
 *  \___|\___/|___/_/_\_\_|_|__|_/_/_\_\_|_ ___ 
 *       / __| | | | _ \ __| _ \___| _ \ __/ __|
 *       \__ \ |_| |  _/ _||   /___|   / _|\__ \
 *       |___/\___/|_| |___|_|_\   |_|_\___|___/
 *                                          2012
 *
 *   by Jens Wetzl           (jens.wetzl@fau.de)
 *  and Oliver Taubmann (oliver.taubmann@fau.de)
 *
 *  This work is licensed under a Creative Commons
 *  Attribution 3.0 Unported License. (CC-BY)
 *  http://creativecommons.org/licenses/by/3.0/
 * 
 **/

#include "cudalbfgs_error_checking.h"

#include <stdio.h>
#include <cmath>

namespace ReductionGPU
{
	template <class T, unsigned int blockSize>
	__global__ void
	sumReduction(const T *g_idata, T *g_odata, const unsigned int width, 
		const unsigned int height, const unsigned int ld);
}

namespace Reduction
{	
	template <class T>
	void sumReduction(const T *d_data, const unsigned int width, const unsigned int height,
		const unsigned int ld, T *d_result, T *d_odata)
	{
		const unsigned int n = width*height;
		
		const int numThreads = 512; // has to be power of 2
		const int numBlocks  = (n % (2*numThreads) == 0)
		                     ?  n / (2*numThreads)
		                     :  n / (2*numThreads) + 1;
		
	    dim3 dimBlock(numThreads, 1, 1);
	    dim3 dimGrid(numBlocks, 1, 1);
		
	    // when there is only one warp per block, we need to allocate two warps 
	    // worth of shared memory so that we don't index shared memory out of bounds
	    int smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);
		
		if (!d_odata)
			CudaSafeCall( cudaMalloc((void**) &d_odata,  numBlocks * sizeof(T)) );
		
		ReductionGPU::sumReduction<T, numThreads><<< dimGrid, dimBlock, smemSize >>>(d_data, d_odata, width, height, ld);
		CudaCheckError();
		cudaDeviceSynchronize();
		
		ReductionGPU::sumReduction<T, numThreads><<< 1, dimBlock, smemSize >>>(d_odata, d_result, numBlocks, 1, numBlocks);
		CudaCheckError();
		cudaDeviceSynchronize();
	}
}

namespace ReductionGPU
{
	// Utility class used to avoid linker errors with extern
	// unsized shared memory arrays with templated type
	template<class T>
	struct SharedMemory
	{
	    __device__ inline operator       T*()
	    {
	        extern __shared__ int __smem[];
	        return (T*)__smem;
	    }

	    __device__ inline operator const T*() const
	    {
	        extern __shared__ int __smem[];
	        return (T*)__smem;
	    }
	};

	// specialize for double to avoid unaligned memory 
	// access compile errors
	template<>
	struct SharedMemory<double>
	{
	    __device__ inline operator       double*()
	    {
	        extern __shared__ double __smem_d[];
	        return (double*)__smem_d;
	    }

	    __device__ inline operator const double*() const
	    {
	        extern __shared__ double __smem_d[];
	        return (double*)__smem_d;
	    }
	};
	
	template <class T, unsigned int blockSize>
	__global__ void
	sumReduction(const T *g_idata, T *g_odata, const unsigned int width, 
		const unsigned int height, const unsigned int ld)
	{
	    T *sdata = SharedMemory<T>();

	    // perform first level of reduction,
	    // reading from global memory, writing to shared memory
	    unsigned int tid = threadIdx.x;
	    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	    unsigned int gridSize = blockSize*2*gridDim.x;
    
		const unsigned int n = width*height;
	
	    T mySum = 0;

	    // we reduce multiple elements per thread.  The number is determined by the 
	    // number of active thread blocks (via gridDim).  More blocks will result
	    // in a larger gridSize and therefore fewer elements per thread
	    while (i < n)
	    {
			const unsigned int y = i / width;
			const unsigned int x = i - y*width;
			
	        mySum += g_idata[y*ld + x];
	        
			// ensure we don't read out of bounds
	        if (i + blockSize < n)
			{
				const unsigned int y2 = (i+blockSize) / width;
				const unsigned int x2 = (i+blockSize) - y2*width;
				
	            mySum += g_idata[y2*ld + x2];
			}
			
	        i += gridSize;
	    } 

	    // each thread puts its local sum into shared memory 
	    sdata[tid] = mySum;
	    __syncthreads();

	    // do reduction in shared mem
	    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
	    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
	    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
    
	    if (tid < 32)
	    {
	        // now that we are using warp-synchronous programming (below)
	        // we need to declare our shared memory volatile so that the compiler
	        // doesn't reorder stores to it and induce incorrect behavior.
	        volatile T* smem = sdata;
	        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
	        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
	        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
	        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
	        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
	        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
	    }
    
	    // write result for this block to global mem 
	    if (tid == 0) 
	        g_odata[blockIdx.x] = sdata[0];
	}
}
