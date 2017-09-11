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

#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <FreeImage.h>
typedef FIBITMAP* ImagePtr;

#include <string>

// Some utility functions for image IO using FreeImage
// (-> http://freeimage.sourceforge.net/).

struct ImageDims
{
	ImageDims(ImagePtr img)
		: width (FreeImage_GetWidth (img))
		, height(FreeImage_GetHeight(img))
	{}

	const size_t width;
	const size_t height;
};

class ImageIO
{

public:

	static ImagePtr loadGrayscaleImage(const char *filename);

	static void saveGPUImage(const std::string &filename, const float *d_image,
							 const size_t width, const size_t height, const size_t ld);
	
	static float *uploadImage(const ImagePtr &img, size_t &width, size_t &height, size_t &ld);

};

#endif // IMAGE_IO_H
