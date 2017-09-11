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

#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

#include "HuberLaplacian.h"
#include "MAPSuperResolution.h"
#include "ImageIO.h"

#include <CudaLBFGS/timer.h>

void printUsage(const char *argv0)
{
	cout << "Usage:   " << argv0 << " lowres-pattern motion-parameters highres-filename=high.png highres-width=400 highres-height=400 psfWidth=0.4" << endl << endl;
	cout << "Example: " << argv0 << " lowres%d.png motion.txt" << endl;
	cout << "         will load lowres0.png, lowres1.png, ..., lowresN.png" << endl <<
			"         until no more images with that filename exist." << endl;
}

int main(int argc, char **argv)
{
#ifdef SUPERRES_TIMING
	timer initTimer("init");
	initTimer.start();
#endif

	// Read command line arguments
	if (argc < 3)
	{
		printUsage(argv[0]);
		return 0;
	}	

	// Required
	const char *lowres_pattern   = argv[1];
	const char *motion_filename  = argv[2];
	
	// Optional
	const char *highres_filename = argc > 3 ? argv[3]               : "high.png";
	const size_t highres_width   = argc > 4 ? size_t(atoi(argv[4])) : 400;
	const size_t highres_height  = argc > 5 ? size_t(atoi(argv[5])) : 400;
	const float psfWidth         = argc > 6 ? atof(argv[6])         : 0.4f;

#ifdef SUPERRES_TIMING
	stringstream ss;
	ss << highres_width << "_" << highres_height << "_" << psfWidth << "_" << 
	      highres_filename << "_";
	timer::timerPrefix = ss.str();
#endif

	// Initialize image loading library

	FreeImage_Initialise();

	// Load low resolution images

	LRImageStack lrImages;
	lrImages.loadFromFiles(lowres_pattern);
	
	// Load motion params

	vector<MotionParams> motionParams;
	MotionParams::loadFromFile(motion_filename, lrImages.getNumImages(), motionParams);
	
#ifdef SUPERRES_TIMING
	initTimer.stop();
	initTimer.saveMeasurement();
	
	timer superresTimer("superres");
	superresTimer.start();
#endif

	// Allocate superresolved image with desired dimensions

	SRImage srImage(highres_height, highres_width);

	// Set up a prior function (optional)

	const bool usePrior = true;

	cost_function *prior = NULL;

	if (usePrior)
		prior = new HuberLaplacian(srImage.getHeight(), srImage.getWidth(), 0.05f, 1000.0f);

	// Create the workhorse object, adjust options to your liking

	MAPSuperResolution mapsr;

	mapsr.setPsfWidth(psfWidth);
	mapsr.setPrior(prior);
	
	// Compute and save the superresolved image

	mapsr.superresolve(lrImages, motionParams, srImage, MAPSuperResolution::SR_INITIALIZATION_AVERAGE);

#ifdef SUPERRES_TIMING
	superresTimer.stop();
	superresTimer.saveMeasurement();
#endif
	
	cout << endl << "Saving result to " << highres_filename << endl;
	srImage.saveToFile(highres_filename);

	// Clean up
	
	srImage.destroy();

	if (usePrior) delete prior;

	FreeImage_DeInitialise();

	return 0;
}
