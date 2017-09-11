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

#include "SRSystemMatrix.h"
#include "cudalbfgs_error_checking.h"

#include <CudaLBFGS/timer.h>

using namespace std;

namespace gpu_SRSystemMatrix
{
	__global__ void composeSingleSystem(const size_t offset, const float *H,
	                                    const size_t lowresWidth,  const size_t lowresHeight,
	                                    const size_t highresWidth, const size_t highresHeight,
	                                    const float psfWidth, const int pixelRadius,
	                                    float *systemMatrixVals, int *systemMatrixCols,
	                                    int *systemMatrixRows);
}


SRSystemMatrix::SRSystemMatrix(const vector<MotionParams> &motionParams, const float psfWidth,
                               const LRImageStack &lrImages, const SRImage &srImage, const GPUHandles &gpuhandles,
                               const float radiusScale)
	: m_height(lrImages.getNumImagePixels() * lrImages.getNumImages())
	, m_width(srImage.getNumPixels())
	, m_psfWidth(psfWidth)
	, m_radiusScale(radiusScale)
	, m_motionParams(motionParams)
	, m_lrImages(lrImages)
	, m_srImage(srImage)
	, m_gpuHandles(gpuhandles)
	, m_d_values_ccs(0)
	, m_d_colPointers_ccs(0)
	, m_d_rowIndices_ccs(0)
{
	compose();
}

SRSystemMatrix::~SRSystemMatrix()
{
	CudaSafeCall( cudaFree(m_d_values)      );
	CudaSafeCall( cudaFree(m_d_rowPointers) );
	CudaSafeCall( cudaFree(m_d_colIndices)  );
	
#ifdef SUPERRES_STORE_TRANSPOSE
	CudaSafeCall( cudaFree(m_d_values_ccs)      );
	CudaSafeCall( cudaFree(m_d_rowIndices_ccs)  );
	CudaSafeCall( cudaFree(m_d_colPointers_ccs) );
#endif
}

void SRSystemMatrix::compose()
{
#ifdef SUPERRES_TIMING
	timer composeTimer("composeSystemMatrix");
	composeTimer.start();
#endif
	
	const float zoom = float(m_srImage.getWidth()) / float(m_lrImages.getImageWidth());
	const float maxPsfRadius = m_radiusScale * zoom * m_psfWidth;
	
	int pixelRadius  = (int)floor(maxPsfRadius + 0.5f);
	
	if (2 * pixelRadius + 1 >= std::min(m_srImage.getWidth(), m_srImage.getHeight()))
	{
		cout << "WARNING: With the chosen settings for radius scale, zoom and psfWidth," <<
		        "the point spread function covers the whole SR image." << endl;
		pixelRadius = (std::min(m_srImage.getWidth(), m_srImage.getHeight()) - 1) / 2;
	}
	
	// Allocate memory for CRS (and CCS)

	// The number of non-zero elements per matrix row is (2 * pixelRadius + 1)^2
	size_t numValElements = (2 * pixelRadius + 1) * (2 * pixelRadius + 1) * m_height;

	CudaSafeCall( cudaMalloc((void**) &m_d_values,      numValElements * sizeof(float)) );
	CudaSafeCall( cudaMalloc((void**) &m_d_colIndices,  numValElements * sizeof(int)  ) );
	CudaSafeCall( cudaMalloc((void**) &m_d_rowPointers, (m_height + 1) * sizeof(int)  ) );
	
	cudaMemset(m_d_colIndices, 0, numValElements * sizeof(int));
	
#ifdef SUPERRES_STORE_TRANSPOSE
	CudaSafeCall( cudaMalloc((void**) &m_d_values_ccs,      numValElements * sizeof(float)) );
	CudaSafeCall( cudaMalloc((void**) &m_d_rowIndices_ccs,  numValElements * sizeof(int)  ) );
	CudaSafeCall( cudaMalloc((void**) &m_d_colPointers_ccs, (m_width + 1)  * sizeof(int)  ) );
#endif

	size_t offset = 0;
	
	// Copy motion parameters to GPU
	
	float *d_motionparams;
	CudaSafeCall( cudaMalloc((void**) &d_motionparams, m_motionParams.size() * 9 * sizeof(float)) );
	CudaSafeCall( cudaMemcpy(d_motionparams, &m_motionParams[0],
	                         m_motionParams.size() * 9 * sizeof(float), cudaMemcpyHostToDevice) );

	// Compose the equation systems for each low-res image
	
	for (size_t i = 0; i < m_lrImages.getNumImages(); ++i)
	{
		composeSingleSystem(offset, i, pixelRadius, d_motionparams);
		offset += m_lrImages.getNumImagePixels();
	}
	
	cudaDeviceSynchronize();

	// The last element of the CRS row pointer is the number of non-zero elements
	// The other entries of the row pointer array are set in the kernel
	CudaSafeCall( cudaMemcpy(m_d_rowPointers + m_height, &numValElements,
							 sizeof(int), cudaMemcpyHostToDevice) );
	
	CudaSafeCall( cudaFree(d_motionparams) );
	
#ifdef SUPERRES_STORE_TRANSPOSE
	// Create CCS structure from CRS structure
	CusparseSafeCall( cusparseScsr2csc(m_gpuHandles.cusparseHandle, m_height, m_width, m_d_values, 
	                                   m_d_rowPointers, m_d_colIndices, m_d_values_ccs, m_d_rowIndices_ccs, 
	                                   m_d_colPointers_ccs, 1, CUSPARSE_INDEX_BASE_ZERO) );
#endif

#ifdef SUPERRES_TIMING
	composeTimer.stop();
	composeTimer.saveMeasurement();
#endif
}

void SRSystemMatrix::composeSingleSystem(const size_t offset, const size_t motionIdx, const int pixelRadius,
										 float *d_motionparams)
{
	size_t height = m_lrImages.getNumImagePixels();

	dim3 blockDim(512);
	dim3 gridDim = dim3(height % blockDim.x == 0 ? height / blockDim.x : (height / blockDim.x) + 1);

	gpu_SRSystemMatrix::composeSingleSystem<<<gridDim, blockDim>>>(offset, &d_motionparams[9 * motionIdx],
	                                                               m_lrImages.getImageWidth(), m_lrImages.getImageHeight(),
	                                                               m_srImage.getWidth(),       m_srImage.getHeight(),
	                                                               m_psfWidth, pixelRadius,
	                                                               m_d_values, m_d_colIndices, m_d_rowPointers);
	CudaCheckError();
}

namespace gpu_SRSystemMatrix
{
	__device__ int roundToInt(float val)
	{
		return (int)floor(val + 0.5f);
	}

	__global__ void composeSingleSystem(const size_t offset, const float *H,
										const size_t lowresWidth,  const size_t lowresHeight,
										const size_t highresWidth, const size_t highresHeight,
										const float psfWidth, const int pixelRadius,
										float *systemMatrixVals, int *systemMatrixCols,
										int *systemMatrixRows)
	{
		const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

		const size_t lowresPixels  = lowresWidth  * lowresHeight;

		if (idx >= lowresPixels)
			return;

		// Coordinates of this thread in the low-res image
		size_t x = idx % lowresWidth;
		size_t y = idx / lowresWidth;

		// Row that this thread writes in the full system matrix
		size_t r = idx + offset;

		// Transform pixel coordinates from the LR grid to the desired HR grid

		float hrx, hry;
		float zoom = float(highresWidth) / float(lowresWidth);

		hrx = (H[0] * x + H[1] * y + H[2]) * zoom;
		hry = (H[3] * x + H[4] * y + H[5]) * zoom;

		float weightSum = 0.0f;

		const size_t maxRowElems = (2 * pixelRadius + 1) * (2 * pixelRadius + 1);
		size_t offsetCRS = 0;
		size_t offsetRows = maxRowElems * r;

		// Iterate over the neighborhood defined by the width of the psf
		for (int offsetY = -pixelRadius; offsetY <= pixelRadius; ++offsetY)
		{
			const int ny = roundToInt(hry + offsetY);

			if (ny < 0 || ny >= highresHeight)
				continue;

			for (int offsetX = -pixelRadius; offsetX <= pixelRadius; ++offsetX)
			{
				const int nx = roundToInt(hrx + offsetX);

				if (nx < 0 || nx >= highresWidth)
					continue;

				const float dx = hrx - float(nx);
				const float dy = hry - float(ny);
				
				// Compute influence of current high-res pixel for 
				// this thread's low-res pixel

				float dist = dx*dx*H[0]*H[0] + dy*dy*H[4]*H[4] +
							 dx*dy*H[0]*H[3] + dx*dy*H[1]*H[4];

				float weight = expf(-dist / (2.0f * zoom * zoom * psfWidth * psfWidth));

				const size_t valIdx = offsetRows + offsetCRS;
				systemMatrixVals[valIdx] = weight;
				systemMatrixCols[valIdx] = ny * highresWidth + nx;

				weightSum += weight;

				++offsetCRS;
			}
		}

		if (weightSum > 0.0f)
		{
			// Normalize row sums
			for (size_t i = 0; i < offsetCRS; ++i)
			{
				systemMatrixVals[offsetRows + i] /= weightSum;
			}
		}
		
		// If we have saved less than maxRowElems elements,
		// we have to pad the CRS structure with 0 entries
		// to make sure it is valid

		if (offsetCRS == 0)
		{
			systemMatrixVals[offsetRows] = 0.0f;
			systemMatrixCols[offsetRows] = 0;
			++offsetCRS;
		}

		bool copy = false;
		
		// Try adding elements after the last saved entry

		while (offsetCRS < maxRowElems)
		{
			const size_t idx = offsetRows + offsetCRS;

			if (systemMatrixCols[idx - 1] + 1 >= highresWidth * highresHeight)
			{
				copy = true;
				break;
			}

			systemMatrixVals[idx] = 0.0f;
			systemMatrixCols[idx] = systemMatrixCols[idx - 1] + 1;
			offsetCRS++;
		}
		
		// If there isn't enough space after the last saved
		// entry, add padding before first entry

		if (copy)
		{
			for (int idx = offsetCRS - 1; idx >= 0; --idx)
			{
				systemMatrixVals[offsetRows + maxRowElems - (offsetCRS - idx)] =
						systemMatrixVals[offsetRows + idx];
				systemMatrixCols[offsetRows + maxRowElems - (offsetCRS - idx)] =
						systemMatrixCols[offsetRows + idx];
			}

			for (int idx = maxRowElems - offsetCRS - 1; idx >= 0; --idx)
			{
				systemMatrixVals[offsetRows + idx] = 0.0f;
				systemMatrixCols[offsetRows + idx] = systemMatrixCols[offsetRows + idx + 1] - 1;
			}
		}

		systemMatrixRows[r] = r * maxRowElems;
	}
}
