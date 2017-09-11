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

#ifndef MOTION_PARAMS_H
#define MOTION_PARAMS_H

#include <string>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

// Stores a homography matrix that transforms a point in the 
// low resolution image to its corresponding point in the 
// coordinate system of the superresolved image.

struct MotionParams
{
	float H[9];
	
	// Loads numImages homography matrices stored in a single file to motionParams
	static void loadFromFile(const std::string &filename, size_t numImages, std::vector<MotionParams> &motionParams)
	{
		std::ifstream file(filename.c_str());

		if (!file.good())
		{
			std::cerr << "Error: Couldn't open '" << filename << "'." << std::endl;
			exit(EXIT_FAILURE);
		}

		motionParams.resize(numImages);

		for (size_t i = 0; i < numImages; ++i)
		{
			MotionParams &mp = motionParams[i];

			try
			{
				file >> mp.H[0] >> mp.H[1] >> mp.H[2] >>
				        mp.H[3] >> mp.H[4] >> mp.H[5] >>
				        mp.H[6] >> mp.H[7] >> mp.H[8];
			}
			catch (...)
			{
				std::cerr << "Error: Couldn't read " << (numImages * 9) <<
					         "floats from motion param file." << std::endl;
				exit(EXIT_FAILURE);
			}
		}
	}
};

#endif // MOTION_PARAMS_H
