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

#ifndef SR_COST_FUNCTION_H
#define SR_COST_FUNCTION_H

#include "LRImageStack.h"
#include "SRSystemMatrix.h"
#include "GPUHandles.h"

#include <CudaLBFGS/cost_function.h>
#include <CudaLBFGS/timer.h>

// The cost function to be minimized.
// Computes the squared Euclidean norm of the
// residual, adding the prior if available.
class SRCostFunction : public cost_function
{
public:
	SRCostFunction(size_t numUnknowns, const SRSystemMatrix &systemMatrix, const LRImageStack &lrImages,
				   const GPUHandles &gpuHandles, cost_function *priorFunction = NULL);

	virtual ~SRCostFunction();

	// Computes function value (d_f) and gradient (d_gradf)
	// at the given positon (d_x), all in device memory.
	void f_gradf(const float *d_x, float *d_f, float *d_gradf);

private:

	float *m_d_residual;
	size_t m_residualSize;

	const SRSystemMatrix &m_systemMatrix;
	const LRImageStack   &m_lrImages;

	const GPUHandles &m_gpuHandles;

	cost_function *m_prior;

	// ---

	timer *m_evalTotalTimer;
	timer *m_evalFTimer;
	timer *m_evalGradTimer;
	timer *m_evalPriorTimer;
};


#endif // SR_COST_FUNCTION_H
