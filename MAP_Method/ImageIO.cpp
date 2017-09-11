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

#include "ImageIO.h"
#include "cudalbfgs_error_checking.h"

#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

ImagePtr ImageIO::loadGrayscaleImage(const char *filename)
{
	ImagePtr image = FreeImage_Load(FIF_PNG, filename);
	ImagePtr imGrayscale;

	FREE_IMAGE_COLOR_TYPE type = FreeImage_GetColorType(image);

	switch(type)
	{
	case FIC_MINISBLACK:
		imGrayscale = image;
		break;
	case FIC_RGB:
	case FIC_RGBALPHA:
		imGrayscale = FreeImage_ConvertToGreyscale(image);
		FreeImage_Unload(image);
		break;
	default:
		cerr << "Error: Image type unsupported" << endl;
		exit(EXIT_FAILURE);
	}
	cout << "Loaded " << filename << "." << endl;

	return imGrayscale;
}

void ImageIO::saveGPUImage(const std::string &filename, const float *d_image,
						   const size_t width, const size_t height, const size_t ld)
{
	const size_t pixels = ld * height;

	float *h_image = new float[pixels];

	CudaSafeCall( cudaMemcpy(h_image, d_image, pixels * sizeof(float),
							 cudaMemcpyDeviceToHost) );
	
	if (filename.substr(filename.size() - 3) == "txt")
	{
		// Output TXT
		
		ofstream file(filename.c_str(), ios_base::out | ios_base::trunc);
		file << setprecision(4);
		
		for (size_t h = 0; h < height; ++h)
		{
			file << h_image[h * ld];
			
			for (size_t w = 1; w < width; ++w)
			{
				file << "," << h_image[h*ld + w];
			}
			
			file << endl;
		}
	}
	else
	{
		// Output PNG
		
		ImagePtr out = FreeImage_Allocate(width, height, 8);
	
		if (!out)
		{
			cerr << "Error: Couldn't allocate output image." << endl;
			exit(EXIT_FAILURE);
		}
	
		for (size_t y = 0; y < height; ++y)
		{
			BYTE  *line  = FreeImage_GetScanLine(out, y);
			float *fline = h_image + (height - y - 1) * ld;
	
			for (size_t x = 0; x < width; ++x)
			{
				int val = (int)floor(fline[x] * 255.0f + 0.5f);
				line[x] = (BYTE)max(0, min(255, val));
			}
		}
	
		FreeImage_Save(FIF_PNG, out, filename.c_str());
		FreeImage_Unload(out);
	}	

	delete [] h_image;
}

float *ImageIO::uploadImage(const ImagePtr &img, size_t &width, size_t &height, size_t &ld)
{
	ImageDims dims(img);
	
	const size_t sz = 8;
	
	width = dims.width;
	height = dims.height;
	
	ld  = (dims.width % sz == 0) ? dims.width : ((dims.width / sz) + 1) * sz;
	// height = (dims.height % sz == 0) ? dims.height : ((dims.height / sz) + 1) * sz;
	
	const size_t numPixels = ld * height;
			
	float *h_img = new float[numPixels]();
			
	for (size_t y = dims.height; y > 0; --y)
	{
		BYTE *line = FreeImage_GetScanLine(img, y-1);
	
		for (size_t x = 0; x < dims.width; ++x)
			h_img[(height-y) * ld + x] = float(line[x]) / 255.0f;
	}
	
	float *d_img;
	CudaSafeCall( cudaMalloc((void**) &d_img, 
	                         numPixels * sizeof(float)) );
	CudaSafeCall( cudaMemcpy(d_img, h_img, numPixels * sizeof(float),
	                         cudaMemcpyHostToDevice) );
	
	delete [] h_img;
	
	return d_img;
}
