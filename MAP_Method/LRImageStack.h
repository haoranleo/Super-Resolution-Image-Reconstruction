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

#ifndef LR_IMAGE_STACK_H
#define LR_IMAGE_STACK_H

#include "ImageIO.h"

#include <string>
#include <vector>

// The series of low resolution images to be superresolved,
// stored as a single vector containing all the pixels in
// row major order.
class LRImageStack
{
public:

	LRImageStack();

	virtual ~LRImageStack();

	// Builds the vector from png files matching the given pattern,
	// e.g. "lr%d.png" tries to load lr0.png, lr1.png, lr2.png, ...
	// until no more images are found.
	// Fixed width patterns such as %03d are also supported.
	void loadFromFiles(const std::string &fileNamePattern);
	
	// Builds the vector from host memory. Each element in images must
	// point to imageWidth * imageHeight contiguous floats, the image
	// pixels in row major order.
	// The input will not be modified, i.e. the caller is responsible
	// for freeing the provided host memory after calling this function.
	void loadFromHostPixels(const size_t imageWidth, const size_t imageHeight,
	                        const std::vector<float*> &images);

	size_t getImageHeight()    const { return m_imageHeight;    }
	size_t getImageWidth ()    const { return m_imageWidth;     }
	size_t getNumImagePixels() const { return m_numImagePixels; }
	size_t getNumImages()      const { return m_numImages;      }

	// Returns a pointer to the low resolution image vector in device memory.
	// It will be freed and no longer valid once the object is destroyed.
	float       *getPixels()       { return m_d_pixels; }
	const float *getPixels() const { return m_d_pixels; }

private:

	size_t m_imageHeight;
	size_t m_imageWidth;
	size_t m_numImagePixels;
	size_t m_numImages;

	float *m_d_pixels;

	void loadFromFilesToVector(const std::string &fileNamePattern, std::vector<float*> &images);
};

#endif // LR_IMAGE_STACK_H
