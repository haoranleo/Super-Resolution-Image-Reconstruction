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

#ifndef SR_IMAGE_H
#define SR_IMAGE_H

#include "LRImageStack.h"

#include "GPUHandles.h"

class SRSystemMatrix; // forward declaration due to cyclic dependency

#include <string>

// The resulting superresolved image to be obtained by optimization.
class SRImage
{
public:

	SRImage(const size_t height, const size_t width);

	virtual ~SRImage();

	size_t getHeight()    const { return m_height;    }
	size_t getWidth()     const { return m_width;     }
	size_t getNumPixels() const { return m_numPixels; }

	// Returns a pointer to the image pixels in device memory, stored in
	// row major order.
	// The pointer will continue to be valid even after this object's
	// destructor is called, so that the final image pixels can be used
	// without this wrapper if desired.
	// This implies that the caller is responsible for calling cudaFree
	// with this pointer once the pixel data is no longer needed, or 
	// alternatively calling destroy in case the object still exists.
	float       *getPixels()       { return m_d_pixels; }
	const float *getPixels() const { return m_d_pixels; }

	// Zeroes out all image pixels, resulting in a black image
	void setZero();

	// Initializes image pixels based on an average of corresponding
	// pixels in the low resolution images based on the model matrix.
	void initToAverageImage(const LRImageStack &lrImages, const SRSystemMatrix &systemMatrix, 
	                        const GPUHandles &gpuHandles);

	// Saves the image to a png file.
	void saveToFile(const std::string &fileName) const;
	
	// Frees the device memory for pixels allocated in the constructor.
	void destroy();

private:

	size_t m_height;
	size_t m_width;
	size_t m_numPixels;

	float *m_d_pixels;
};

#endif // SR_IMAGE_H
