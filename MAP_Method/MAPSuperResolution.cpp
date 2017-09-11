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

#include "MAPSuperResolution.h"

#include "SRSystemMatrix.h"
#include "SRCostFunction.h"

#include <CudaLBFGS/lbfgs.h>

#include <iostream>
using namespace std;

MAPSuperResolution::MAPSuperResolution()
	: m_prior(NULL)
	, m_psfWidth(0.4f)
	, m_gradientEps(1e-13f)
{
}

MAPSuperResolution::~MAPSuperResolution()
{
}

void MAPSuperResolution::superresolve(const LRImageStack &lrImages, const std::vector<MotionParams> &motionParams,
									  SRImage &srImage, MAPSuperResolution::SRInitialization init)
{
	// Compute system matrix

	cout << endl << "Computing system matrix..." << endl;
	SRSystemMatrix systemMatrix(motionParams, m_psfWidth, lrImages, srImage, m_gpuHandles);

	// Initalize SR Image

	switch (init)
	{
		case SR_INITIALIZATION_AVERAGE:
			srImage.initToAverageImage(lrImages, systemMatrix, m_gpuHandles);
			break;
		case SR_INITIALIZATION_BLACK:
			srImage.setZero();
			break;
	}

	// Optimize

	cout << endl << "Optimizing..." << endl;
	SRCostFunction cf(srImage.getNumPixels(), systemMatrix, lrImages, m_gpuHandles, m_prior);

	lbfgs minimizer(cf);
	minimizer.setGradientEpsilon(m_gradientEps);
	minimizer.setMaxIterations(3);

	lbfgs::status stat = minimizer.minimize(srImage.getPixels());

	cout << "Optimization result: " << lbfgs::statusToString(stat) << endl;
}
