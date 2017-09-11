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

#ifndef SR_SYSTEM_MATRIX_H
#define SR_SYSTEM_MATRIX_H

#include "LRImageStack.h"
#include "SRImage.h"
#include "MotionParams.h"
#include "GPUHandles.h"

#include <vector>

// The m by n model matrix representing the
// transformation of the ideal high resolution
// image to the series of given low resolution
// images.
//
// m = #(pixels per lr image) * #(lr images)
// n = #(pixels in  hr image)
//
// The matrix is stored in CRS (compressed row
// storage) format, and -- if the approriate
// build option was chosen -- additionally in
// CCS (compressed column storage) format.
class SRSystemMatrix
{
public:

	// Builds the matrix.
	SRSystemMatrix(const std::vector<MotionParams> &motionParams, const float psfWidth,
	               const LRImageStack &lrImages, const SRImage &srImage, const GPUHandles &gpuhandles, 
	               const float radiusScale = 3.0f);

	virtual ~SRSystemMatrix();

	size_t getHeight() const { return m_height; }
	size_t getWidth () const { return m_width;  }

	// Internal data (CRS format).
	float *getValues()      { return m_d_values;      }
	int   *getRowPointers() { return m_d_rowPointers; }
	int   *getColIndices()  { return m_d_colIndices;  }

	const float *getValues()      const { return m_d_values;      }
	const int   *getRowPointers() const { return m_d_rowPointers; }
	const int   *getColIndices()  const { return m_d_colIndices;  }
	
	// Internal data (CCS format).
	// Only available if SUPERRES_STORE_TRANSPOSE was chosen.
	float *getValuesCCS()      { return m_d_values_ccs;      }
	int   *getRowIndicesCCS()  { return m_d_rowIndices_ccs;  }
	int   *getColPointersCCS() { return m_d_colPointers_ccs; }
	
	const float *getValuesCCS()      const { return m_d_values_ccs;      }
	const int   *getRowIndicesCCS()  const { return m_d_rowIndices_ccs;  }
	const int   *getColPointersCCS() const { return m_d_colPointers_ccs; }

private:

	size_t m_height;
	size_t m_width;
	
	float m_psfWidth;    // Width of the point spread function
	float m_radiusScale; // Scaling factor for the size of the neighbourhood 
	                     // in the high resolution image to be considered

	const std::vector<MotionParams> m_motionParams;
	const LRImageStack &m_lrImages;
	const SRImage &m_srImage;
	const GPUHandles &m_gpuHandles;

	float *m_d_values;
	int   *m_d_rowPointers;
	int   *m_d_colIndices;
	
	float *m_d_values_ccs;
	int   *m_d_rowIndices_ccs;
	int   *m_d_colPointers_ccs;

	void compose();
	void composeSingleSystem(const size_t offset, const size_t motionIdx, const int pixelRadius,
	                         float *d_motionparams);
};

#endif // SR_SYSTEM_MATRIX_H
