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

#ifndef MAP_SUPER_RESOLUTION_H
#define MAP_SUPER_RESOLUTION_H

#include "LRImageStack.h"
#include "MotionParams.h"
#include "SRImage.h"
#include "GPUHandles.h"

#include <CudaLBFGS/cost_function.h>

#include <vector>

// The workhorse class of this application.
// Performs Maximum a Posteriori Superresolution
// given a series of low resolution imagers, motion
// and camera parameters.
class MAPSuperResolution
{
public:

	// How to initialize the superresolved image.
	enum SRInitialization {
		SR_INITIALIZATION_AVERAGE = 0, // Build an "average image" from the low resolution images
		SR_INITIALIZATION_BLACK,       // Initialize to zeros (= a black image)
		SR_INITIALIZATION_KEEP         // Do not modify the image provided by the caller before optimization
	};

	MAPSuperResolution();

	virtual ~MAPSuperResolution();

	// Performs all steps required to create a superresolved image from a series of low
	// resolution images and corresponding motion parameters.
	// See the enum above for available initialization options.
	void superresolve(const LRImageStack &lrImages, const std::vector<MotionParams> &motionParams,
					  SRImage &srImage, SRInitialization init = SR_INITIALIZATION_AVERAGE);

	// Allows to set/get the prior function to be used which must implement
	// the cost_function interface of CudaLBFGS (combined evaluation of function
	// value and gradient).
	// If no prior (or a NULL pointer) is set, ML (maximum likelihood) estimation
	// will be performed instead of MAP.
	void           setPrior(cost_function *prior) { m_prior = prior; }
	cost_function *getPrior()                     { return m_prior;  }

	void  setPsfWidth(const float psfWidth) { m_psfWidth = psfWidth; }
	float getPsfWidth()                     { return m_psfWidth;     }

	void  setGradientEpsilon(const float gradientEps) { m_gradientEps = gradientEps; }
	float getGradientEpsilon()                        { return m_gradientEps;        }

private:

	GPUHandles     m_gpuHandles;  // cuBLAS / cuSPARSE state

	cost_function *m_prior;       // see set/getPrior()

	float          m_psfWidth;    // Width of the point spread function
	float          m_gradientEps; // Gradient epsilon for the optimization step,
	                              // cf. CudaLBFGS/lbfgs.h for details
};

#endif // MAP_SUPER_RESOLUTION_H
