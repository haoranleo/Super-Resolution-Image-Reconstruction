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

#ifndef GPU_HANDLES_H
#define GPU_HANDLES_H

#include "cudalbfgs_error_checking.h"

// A small wrapper for cuBLAS and cuSPARSE states with
// automatic initialization and cleanup.
struct GPUHandles
{
	cublasHandle_t     cublasHandle;
	cusparseHandle_t   cusparseHandle;
	cusparseMatDescr_t cusparseDescriptor;

	GPUHandles()
	{
		CublasSafeCall  ( cublasCreate          (&cublasHandle)       );
		CusparseSafeCall( cusparseCreate        (&cusparseHandle)     );
		CusparseSafeCall( cusparseCreateMatDescr(&cusparseDescriptor) );

		CusparseSafeCall( cusparseSetMatType     (cusparseDescriptor, CUSPARSE_MATRIX_TYPE_GENERAL) );
		CusparseSafeCall( cusparseSetMatIndexBase(cusparseDescriptor, CUSPARSE_INDEX_BASE_ZERO)     );
	}

	~GPUHandles()
	{
		CublasSafeCall  ( cublasDestroy          (cublasHandle)       );
		CusparseSafeCall( cusparseDestroy        (cusparseHandle)     );
		CusparseSafeCall( cusparseDestroyMatDescr(cusparseDescriptor) );
	}
};

#endif // GPU_HANDLES_H
