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

#ifndef HUBERLOG_H
#define HUBERLOG_H

#include <CudaLBFGS/cost_function.h>
#include <CudaLBFGS/timer.h>

// A prior function on the high resolution image.
// Filters the image using a discrete Laplacian, then
// computes the pseudo-Huber loss function on the result.
class HuberLaplacian : public cost_function
{
public:
	HuberLaplacian(const size_t height, const size_t width,
	         const float alpha = 0.05f, const float strength = 5.0f);

	virtual ~HuberLaplacian();

	// Computes function value (d_f) and gradient (d_gradf)
	// at the given positon (d_x), all in device memory.
	virtual void f_gradf(const float *d_x, float *d_f, float *d_gradf);

private:
	
	size_t m_width;   // Width  of the image
	size_t m_height;  // Height of the image

	float m_alpha;    // Huber threshold
	float m_strength; // Prior strength (weight in the cost function)
	
	timer *m_atomic;
	timer *m_filter;
	
	float *m_reductionArray;
	float *m_reductionArray2;
};

#endif // HUBERLOG_H
