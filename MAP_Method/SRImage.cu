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

#include "SRImage.h"

#include "SRSystemMatrix.h"
#include "ImageIO.h"
#include "cudalbfgs_error_checking.h"

#include <CudaLBFGS/timer.h>

namespace gpu_SRImage
{
	__global__ void computeCSRColSums(float *d_colSums, const float *d_systemMatrixVals,
								      const int *d_systemMatrixRows, const int *d_systemMatrixCols,
								      const size_t m, const size_t n);

	__global__ void elementwiseDiv(float *a, const float *b, const size_t len);
	
	__global__ void divideByCSCColSums(const float *values, const int *colPointers, 
	                                   float *pixels, const size_t n);
}

SRImage::SRImage(const size_t height, const size_t width)
	: m_height(height)
	, m_width(width)
	, m_numPixels(height * width)
{
	CudaSafeCall( cudaMalloc((void**) &m_d_pixels, m_numPixels * sizeof(float)) );
}

SRImage::~SRImage()
{
}

void SRImage::setZero()
{
	CudaSafeCall( cudaMemset(m_d_pixels, 0, m_numPixels * sizeof(float)) );
}

void SRImage::initToAverageImage(const LRImageStack &lrImages, const SRSystemMatrix &systemMatrix, 
                                 const GPUHandles &gpuHandles)
{
#ifdef SUPERRES_TIMING
	timer averageTimer("averageImage");
	timer colSums("avgColSums");
	averageTimer.start();
#endif
	
	const size_t m = systemMatrix.getHeight();
	const size_t n = systemMatrix.getWidth();

	// Compute highresImage = systemMatrix^T lowresVector

#ifndef SUPERRES_STORE_TRANSPOSE
	// Use CRS for transpose-multiply (inefficient)
	CusparseSafeCall( cusparseScsrmv(gpuHandles.cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, m, n, 1.0f, 
	                                 gpuHandles.cusparseDescriptor, systemMatrix.getValues(), 
	                                 systemMatrix.getRowPointers(), systemMatrix.getColIndices(),
	                                 lrImages.getPixels(), 0.0f, m_d_pixels) );
#else
	// Use CCS for transpose-multiply
	CusparseSafeCall( cusparseScsrmv(gpuHandles.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, m, 1.0f, 
	                                 gpuHandles.cusparseDescriptor, systemMatrix.getValuesCCS(), 
	                                 systemMatrix.getColPointersCCS(), systemMatrix.getRowIndicesCCS(),
	                                 lrImages.getPixels(), 0.0f, m_d_pixels) );
#endif
	CudaCheckError();
	cudaDeviceSynchronize();

	// Compute column sums of the system matrix to colSums

#ifndef SUPERRES_STORE_TRANSPOSE
	float *d_colSums;
	CudaSafeCall( cudaMalloc((void**) &d_colSums, n * sizeof(float)) );
	CudaSafeCall( cudaMemset(d_colSums, 0, n * sizeof(float)) );
	
	{
		dim3 blockDim(512);
		dim3 gridDim = (m % blockDim.x == 0) ? (m / blockDim.x)
											 : (m / blockDim.x) + 1;

		gpu_SRImage::computeCSRColSums<<<gridDim, blockDim>>>(d_colSums, systemMatrix.getValues(), systemMatrix.getRowPointers(),
		                                                      systemMatrix.getColIndices(), m, n);

		CudaCheckError();
		cudaDeviceSynchronize();
	}

	// Compute pixels[i] /= colSums[i], i = 0..n-1

	{
		dim3 blockDim(512);
		dim3 gridDim = (n % blockDim.x == 0) ? (n / blockDim.x)
										 : (n / blockDim.x) + 1;

		gpu_SRImage::elementwiseDiv<<<gridDim, blockDim>>>(m_d_pixels, d_colSums, n);
		CudaCheckError();
		cudaDeviceSynchronize();
	}
	
	CudaSafeCall( cudaFree(d_colSums) );
#else
	dim3 blockDim(512);
	dim3 gridDim = (n % blockDim.x == 0) ? (n / blockDim.x)
										 : (n / blockDim.x) + 1;
	
	gpu_SRImage::divideByCSCColSums<<<gridDim, blockDim>>>(systemMatrix.getValuesCCS(), 
	                                                       systemMatrix.getColPointersCCS(), m_d_pixels, n);
#endif
	
#ifdef SUPERRES_TIMING
	averageTimer.stop();
	averageTimer.saveMeasurement();
#endif

//	saveToFile("highres_initial.txt");
}


void SRImage::saveToFile(const std::string &fileName) const
{
	ImageIO::saveGPUImage(fileName, m_d_pixels, m_width, m_height, m_width);
}

void SRImage::destroy()
{
	CudaSafeCall( cudaFree(m_d_pixels) );
}

namespace gpu_SRImage
{
	__device__ static void myAtomicAdd(float *address, float value)
	{
#if __CUDA_ARCH__ >= 200
		atomicAdd(address, value);
#else
		// cf. https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks
		int oldval, newval, readback;

		oldval = __float_as_int(*address);
		newval = __float_as_int(__int_as_float(oldval) + value);
		while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval)
		{
			oldval = readback;
			newval = __float_as_int(__int_as_float(oldval) + value);
		}
#endif
	}
	
	__global__ void computeCSRColSums(float *d_colSums, const float *d_systemMatrixVals,
								      const int *d_systemMatrixRows, const int *d_systemMatrixCols,
								      const size_t m, const size_t n)
	{
		const size_t row = blockIdx.x * blockDim.x + threadIdx.x;

		if (row >= m)
			return;

		for (size_t cidx = d_systemMatrixRows[row]; cidx < d_systemMatrixRows[row+1]; ++cidx)
		{
			myAtomicAdd(d_colSums + d_systemMatrixCols[cidx], d_systemMatrixVals[cidx]);
		}
	}

	__global__ void elementwiseDiv(float *a, const float *b, const size_t len)
	{
		const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx >= len)
			return;

		a[idx] /= b[idx] + 1e-6f;
	}
	
	__global__ void divideByCSCColSums(const float *values, const int *colPointers, 
	                                   float *pixels, const size_t n)
	{
		const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		
		if (idx >= n)
			return;
		
		float weight = 0.0f;
		
		for (size_t ridx = colPointers[idx]; ridx < colPointers[idx+1]; ++ridx)
		{
			weight += values[ridx];
		}
		
		pixels[idx] /= weight + 1e-6f;
	}
}
