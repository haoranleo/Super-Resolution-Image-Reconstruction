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

#include "HuberLaplacian.h"
#include "Reduction.h"

#include "CudaLBFGS/error_checking.h"

namespace gpu_HuberLaplacian
{
	__global__ void laplacian(float *dst, const float *src, const size_t width, const size_t height, 
	                    const size_t pixelsPerThread);
	__global__ void huber(float *x, const size_t width, const size_t height, const float alpha, 
	                      const float strength, const size_t pixelsPerThread, float *f);

	float *d_tmp;
}

HuberLaplacian::HuberLaplacian(const size_t height, const size_t width,
                               const float alpha, const float strength)
   : cost_function(height * width)
   , m_height(height)
   , m_width(width)
   , m_alpha(alpha)
   , m_strength(strength)
{
	CudaSafeCall( cudaMalloc(&gpu_HuberLaplacian::d_tmp, m_numDimensions * sizeof(float)) );
	CudaSafeCall( cudaMalloc( (void**) &m_reductionArray,  width * height * sizeof(float)) );
	CudaSafeCall( cudaMalloc( (void**) &m_reductionArray2, 1024 * sizeof(float)) );
	
#ifdef SUPERRES_TIMING
	m_atomic = new timer("priorOther");
	m_filter = new timer("priorFilter");
#endif
}

HuberLaplacian::~HuberLaplacian()
{
	CudaSafeCall( cudaFree(gpu_HuberLaplacian::d_tmp) );
	CudaSafeCall( cudaFree(m_reductionArray)    );
	CudaSafeCall( cudaFree(m_reductionArray2)   );
	
#ifdef SUPERRES_TIMING
	m_atomic->saveMeasurement();
	m_filter->saveMeasurement();
	
	delete m_atomic;
	delete m_filter;
#endif
}

void HuberLaplacian::f_gradf(const float *d_x, float *d_f, float *d_gradf)
{
	using namespace gpu_HuberLaplacian;

	dim3 blockDim(512);
	
	const size_t pixelsPerThread = 8;
	size_t threadsPerColumn = (m_height % pixelsPerThread == 0) ? (m_height / pixelsPerThread)
	                                                            : (m_height / pixelsPerThread) + 1;
	size_t threads = threadsPerColumn * m_width;
	
	dim3 gridDim  = (threads % blockDim.x == 0) ? (threads / blockDim.x)
		                                        : (threads / blockDim.x) + 1;
	
#ifdef SUPERRES_TIMING
	m_filter->start();
#endif

	// Compute image Laplacian
	laplacian<<<gridDim, blockDim>>>(d_tmp, d_x, m_width, m_height, pixelsPerThread);

	CudaCheckError();
	cudaDeviceSynchronize();
	
#ifdef SUPERRES_TIMING
	m_filter->stop();
	m_atomic->start();
#endif

	CudaSafeCall( cudaMemset(m_reductionArray, 0, m_width * m_height * sizeof(float)) );
	
	// Compute prior function value and gradient without final filtering
	huber<<<gridDim, blockDim>>>(d_tmp, m_width, m_height, m_alpha, m_strength, pixelsPerThread, m_reductionArray);

	CudaCheckError();
	cudaDeviceSynchronize();
	
	Reduction::sumReduction(m_reductionArray, m_width, m_height, m_width, d_f, m_reductionArray2);
	
#ifdef SUPERRES_TIMING
	m_atomic->stop();
	m_filter->start();
#endif

	// Compute Laplacian of the gradient
	laplacian<<<gridDim, blockDim>>>(d_gradf, d_tmp, m_width, m_height, pixelsPerThread);

	CudaCheckError();
	cudaDeviceSynchronize();
	
#ifdef SUPERRES_TIMING
	m_filter->stop();
#endif
}


namespace gpu_HuberLaplacian
{

	__global__ void laplacian(float *dst, const float *src, const size_t width, const size_t height, 
	                    const size_t pixelsPerThread)
	{
		const size_t col  = (blockIdx.x * blockDim.x + threadIdx.x) % width;
		const size_t crow = (blockIdx.x * blockDim.x + threadIdx.x) / width * pixelsPerThread;
		
		if (col >= width || crow >= height)
			return;

		const size_t srow = crow + 1;
		const size_t erow = min((unsigned int)(crow + pixelsPerThread - 1), (unsigned int)(height - 1));
		
		// First element

		const size_t firstIdx = crow * width + col;

		dst[firstIdx] = src[firstIdx];
		
		if (crow + 1 <  height) dst[firstIdx] -= 0.25f * src[firstIdx + width]; // S
		if (crow     >= 1)      dst[firstIdx] -= 0.25f * src[firstIdx - width]; // N
		if (col + 1  <  width)  dst[firstIdx] -= 0.25f * src[firstIdx + 1]; // E
		if (col      >= 1)      dst[firstIdx] -= 0.25f * src[firstIdx - 1]; // W

		// Inner elements

		for (int row = srow; row < erow; ++row)
		{
			const size_t cIdx = row * width + col;
		
			// C, S, N (always exist)
			dst[cIdx] = src[cIdx] - 0.25f * (src[cIdx + width] + src[cIdx - width]);

			if (col + 1 < width) dst[cIdx] -= 0.25f * src[cIdx + 1]; // E
			if (col     >= 1)    dst[cIdx] -= 0.25f * src[cIdx - 1]; // W
		}
		
		if (erow <= crow)
			return;

		// Last element

		const size_t lastIdx = erow * width + col;

		dst[lastIdx] = src[lastIdx] - 0.25f * src[lastIdx - width]; // C, N
		
		if (erow + 1 <  height) dst[lastIdx] -= 0.25f * src[lastIdx + width]; // S
		if (col + 1  <  width)  dst[lastIdx] -= 0.25f * src[lastIdx + 1]; // E
		if (col      >= 1)      dst[lastIdx] -= 0.25f * src[lastIdx - 1]; // W
	}

	__global__ void huber(float *a, const size_t width, const size_t height, const float alpha, 
	                      const float strength, const size_t pixelsPerThread, float *f)
	{
		const size_t col  = (blockIdx.x * blockDim.x + threadIdx.x) % width;
		const size_t crow = (blockIdx.x * blockDim.x + threadIdx.x) / width * pixelsPerThread;
		
		if (col >= width || crow >= height)
			return;

		const size_t erow = min((unsigned int)(crow + pixelsPerThread), (unsigned int)height);

		const float alpha2 = alpha * alpha;

		float colF = 0.0f;

		for (size_t row = crow; row < erow; ++row)
		{
			const size_t idx = row * width + col;
		
			// Pseudo-Huber loss function
			const float root = sqrtf(1.0f + a[idx]*a[idx] / alpha2); 
			colF += alpha2 * (root - 1.0f);
			a[idx] *= strength / root;
		}

		colF *= strength;
		f[blockIdx.x * blockDim.x + threadIdx.x] = colF;
	}

}
