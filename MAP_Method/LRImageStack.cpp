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

#include "LRImageStack.h"
#include "cudalbfgs_error_checking.h"

#include <CudaLBFGS/timer.h>

#include <fstream>
#include <sstream>
#include <limits>

using namespace std;

LRImageStack::LRImageStack()
	: m_d_pixels(0)
{}

LRImageStack::~LRImageStack()
{
	if (m_d_pixels != 0)
		CudaSafeCall( cudaFree(m_d_pixels) );
}

void LRImageStack::loadFromFiles(const std::string &fileNamePattern)
{
	vector<float*> images;

	loadFromFilesToVector(fileNamePattern, images);
	loadFromHostPixels(m_imageWidth, m_imageHeight, images);
	
	for (size_t i = 0; i < images.size(); ++i)
		delete [] images[i];
}

void LRImageStack::loadFromHostPixels(const size_t imageWidth, const size_t imageHeight,
                                      const vector<float*> &h_images)
{
	m_imageWidth     = imageWidth;
	m_imageHeight    = imageHeight;
	m_numImages      = h_images.size();
	m_numImagePixels = imageWidth * imageHeight;
	
#ifdef SUPERRES_TIMING
	timer uploadTimer("imageUpload");
	uploadTimer.start();
#endif
	
	const size_t numPixels = m_numImagePixels * m_numImages;

	if (m_d_pixels != 0)
		CudaSafeCall( cudaFree(m_d_pixels) );
	
	CudaSafeCall( cudaMalloc((void**) &m_d_pixels, numPixels * sizeof(float)) );

	size_t offset = 0;
	vector<float*>::const_iterator it;

	for (it = h_images.begin(); it != h_images.end(); ++it)
	{
		const float *img = *it;

		CudaSafeCall( cudaMemcpy(m_d_pixels + offset, img,
		                         m_numImagePixels * sizeof(float),
		                         cudaMemcpyHostToDevice) );

		offset += m_numImagePixels;
	}

#ifdef SUPERRES_TIMING
	uploadTimer.stop();
	uploadTimer.saveMeasurement();
#endif
}

void LRImageStack::loadFromFilesToVector(const std::string &fileNamePattern, vector<float*> &images)
{
	size_t percentPos = fileNamePattern.find("%");

	if (percentPos == string::npos)
	{
		cerr << "Error: lowres-pattern must contain format string like %[0X]d" << endl;
		exit(EXIT_FAILURE);
	}

	size_t dPos = fileNamePattern.find("d", percentPos);

	if (dPos == string::npos)
	{
		cerr << "Error: lowres-pattern must contain format string like %[0X]d" << endl;
		exit(EXIT_FAILURE);
	}

	bool is_txt = ".txt" == fileNamePattern.substr(fileNamePattern.size() - 4, 4);

	char buf[4096];

	for (int i = 0;; ++i)
	{
		sprintf(buf, fileNamePattern.c_str(), i);

		ifstream file(buf);

		if (file)
		{
			float *h_img = NULL;

			if (is_txt) 
			{
				// Parse text file
				// ---------------

				//cout << "Reading in txt mode" << endl;

				vector<float> tmp_pixels;
				tmp_pixels.reserve(200*200);

				size_t height = 0;
				size_t width  = 0;

				float maxVal = 0.0f, minVal = numeric_limits<float>::max();

				string line;
				while (getline(file, line))
				{
					size_t curWidth = 0;

					stringstream line_stream(line);

					char comma;
					float val;
					while (line_stream >> val)
					{
						++curWidth;
						tmp_pixels.push_back(val);

						if (val > maxVal)
							maxVal = val;
						if (val < minVal)
							minVal = val;

						line_stream >> comma;
					}

					if (width == 0)
						width = curWidth;

					if (width != curWidth)
					{
						cerr << "Error: All image rows must have the same number of pixels" << endl;
						exit(EXIT_FAILURE);
					}

					++height;
				}

				if (i == 0)
				{
					m_imageWidth  = width;
					m_imageHeight = height;
				}
				else if (width  != m_imageWidth ||
				         height != m_imageHeight)
				{
					cerr << "Error: all lowres images must have the same size" << endl;
					exit(EXIT_FAILURE);
				}

				h_img = new float[height * width];

				// TODO: Normalize to [0..1] as done below? Any other normalization?

				copy(tmp_pixels.begin(), tmp_pixels.end(), h_img);

//				for (size_t n = 0; n < height*width; ++n)
//				{
//					h_img[n] = (tmp_pixels[n] - minVal) / (maxVal - minVal);
//				}

				//cout << "Loaded image with dims " << width << "x" << height << endl;

			}
			else
			{
				// Let ImageIO do the loading
				// --------------------------
				ImagePtr img = ImageIO::loadGrayscaleImage(buf);

				ImageDims dims(img);

				if (i == 0)
				{
					m_imageWidth  = dims.width;
					m_imageHeight = dims.height;
				}
				else if (dims.width  != m_imageWidth ||
				         dims.height != m_imageHeight)
				{
					cerr << "Error: all lowres images must have the same size" << endl;
					exit(EXIT_FAILURE);
				}

				h_img = new float[m_imageWidth * m_imageHeight];

				for (size_t y = m_imageHeight; y > 0; --y)
				{
					BYTE *line = FreeImage_GetScanLine(img, y-1);

					for (size_t x = 0; x < m_imageWidth; ++x)
						h_img[(m_imageHeight-y) * m_imageWidth + x] = float(line[x]) / 255.0f;
				}

				FreeImage_Unload(img);
			}

			// Add image to sequence
			images.push_back(h_img);
		}
		else
			break;
	}

	if (images.size() == 0)
	{
		cerr << "Error: No lowres images match the given pattern." << endl;
		exit(EXIT_FAILURE);
	}
}
