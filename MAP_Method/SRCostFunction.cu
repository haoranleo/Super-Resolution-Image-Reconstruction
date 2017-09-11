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

#include "SRCostFunction.h"

#include "ImageIO.h"

#include <iostream>
#include "cudalbfgs_error_checking.h"

using namespace std;

namespace gpu_SRCostFunction
{
	__device__ float d_priorF;

	__global__ void add(float *p, float *q) { *p += *q; }
}

SRCostFunction::SRCostFunction(size_t numUnknowns, const SRSystemMatrix &systemMatrix, 
                               const LRImageStack &lrImages, const GPUHandles &gpuHandles, 
                               cost_function *priorFunction)
	: cost_function(numUnknowns)
	, m_systemMatrix(systemMatrix)
	, m_lrImages(lrImages)
	, m_gpuHandles(gpuHandles)
	, m_prior(priorFunction)
	, m_residualSize(lrImages.getNumImagePixels() * lrImages.getNumImages())
{
	CudaSafeCall( cudaMalloc((void**) &m_d_residual, m_residualSize * sizeof(float)) );
	
#ifdef SUPERRES_TIMING
	m_evalTotalTimer = new timer("evalTotal");
	m_evalFTimer     = new timer("evalFTotal");
	m_evalGradTimer  = new timer("evalGradTotal");
	m_evalPriorTimer = new timer("evalPriorTotal");
#endif
}

SRCostFunction::~SRCostFunction()
{
#ifdef SUPERRES_TIMING
	m_evalTotalTimer->saveMeasurement();
	m_evalFTimer->saveMeasurement();
	m_evalGradTimer->saveMeasurement();
	m_evalPriorTimer->saveMeasurement();
	
	delete m_evalTotalTimer;
	delete m_evalFTimer;
	delete m_evalGradTimer;
	delete m_evalPriorTimer;
#endif
	
	CudaSafeCall( cudaFree(m_d_residual) );
}

void SRCostFunction::f_gradf(const float *d_x, float *d_f, float *d_gradf)
{
	/**
	  * Mathematical notation:
	  * W: Super-resolution system matrix
	  * x: Super-resolution image
	  * y: Low resolution image stack
	  *
	  * f: Objective function value
	  * grad: Objective function gradient
	  */
	
#ifdef SUPERRES_TIMING
	timer evalTimer("eval");
	evalTimer.start();
	m_evalTotalTimer->start();
	m_evalFTimer->start();
#endif
	
//	static int evals = 0;
//	char buf[4096];
//	sprintf(buf, "high_%04d.png", evals++);
//	ImageIO::saveGPUImage(std::string(buf), d_x, 512, 512);

	// Compute residual

	CudaSafeCall( cudaMemcpy(m_d_residual, m_lrImages.getPixels(),
	                         m_residualSize * sizeof(float),
	                         cudaMemcpyDeviceToDevice) );

	// r = W*x - y
	CusparseSafeCall( cusparseScsrmv(m_gpuHandles.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
	                                 m_residualSize, m_numDimensions, 1.0f, m_gpuHandles.cusparseDescriptor,
	                                 m_systemMatrix.getValues(), m_systemMatrix.getRowPointers(), 
	                                 m_systemMatrix.getColIndices(), d_x, -1.0f, m_d_residual) );
	
	CudaCheckError();
	cudaDeviceSynchronize();

	// Compute squared norm of the residual

	CublasSafeCall( cublasSetPointerMode(m_gpuHandles.cublasHandle, CUBLAS_POINTER_MODE_DEVICE) );

	// f = r^T * r
	CublasSafeCall( cublasSdot(m_gpuHandles.cublasHandle, m_residualSize, m_d_residual, 1, m_d_residual, 1, d_f) );

	CudaCheckError();
	cudaDeviceSynchronize();

#ifdef SUPERRES_TIMING
	m_evalFTimer->stop();
	m_evalPriorTimer->start();
#endif
	
	// Add prior, compute prior gradient in d_gradf
	
	float fval;
	cudaMemcpy(&fval, d_f, sizeof(float), cudaMemcpyDeviceToHost);

	if (m_prior)
	{
		float *d_priorFPtr;
		CudaSafeCall( cudaGetSymbolAddress((void**) &d_priorFPtr, gpu_SRCostFunction::d_priorF) );

		// grad = grad_prior
		m_prior->f_gradf(d_x, d_priorFPtr, d_gradf);

		// f += f_prior
		gpu_SRCostFunction::add<<<1, 1>>>(d_f, d_priorFPtr);
		
		CudaCheckError();
		cudaDeviceSynchronize();
	}
	else
		CudaSafeCall( cudaMemset(d_gradf, 0, m_numDimensions * sizeof(float)) );

#ifdef SUPERRES_TIMING
	m_evalPriorTimer->stop();
	m_evalGradTimer->start();
#endif
	
	// Total gradient

	// grad += W^T * r
#ifndef SUPERRES_STORE_TRANSPOSE
	// Use CCS for transpose-multiply
	CusparseSafeCall( cusparseScsrmv(m_gpuHandles.cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
	                                 m_residualSize, m_numDimensions, 2.0f, m_gpuHandles.cusparseDescriptor,
	                                 m_systemMatrix.getValues(), m_systemMatrix.getRowPointers(), 
	                                 m_systemMatrix.getColIndices(), m_d_residual, 1.0f, d_gradf) );
#else
	// Use CRS (inefficient)
	CusparseSafeCall( cusparseScsrmv(m_gpuHandles.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
	                                 m_numDimensions, m_residualSize, 2.0f, m_gpuHandles.cusparseDescriptor,
	                                 m_systemMatrix.getValuesCCS(), m_systemMatrix.getColPointersCCS(), 
	                                 m_systemMatrix.getRowIndicesCCS(), m_d_residual, 1.0f, d_gradf) );
#endif

	CudaCheckError();
	cudaDeviceSynchronize();

#ifdef SUPERRES_TIMING
	m_evalGradTimer->stop();
	m_evalTotalTimer->stop();
	evalTimer.stop();
	evalTimer.saveMeasurement();
#endif
}
