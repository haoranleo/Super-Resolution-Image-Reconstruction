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

#ifndef REDUCTION_H
#define REDUCTION_H

// A helper for performing parallel reduction on the GPU.
namespace Reduction
{	
	// Computes the sum over all elements in a matrix
	// stored linearly in d_data with the given width,
	// height and leading dimension (ld) to d_result.
	// Allocates temporary storage on the device unless 
	// it is provided by the caller in d_tmp.
	template <class T>
	void sumReduction(const T *d_data, const unsigned int width, const unsigned int height,
		const unsigned int ld, T *d_result, T *d_tmp = 0);
}

#include "Reduction.cu"

#endif // REDUCTION_H
