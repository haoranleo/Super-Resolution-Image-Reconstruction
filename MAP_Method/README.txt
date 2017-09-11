NOTE: This library was only tested with CUDA 4.x and
5.x and may not work with more recent versions. We
do not currently have the time to update it for more
recent CUDA versions, but would gladly accept pull
requests addressing this issue.

===========================================================
        ___ _   _ ___   _     __  __   _   ___     
       / __| | | |   \ /_\   |  \/  | /_\ | _ \    
      | (__| |_| | |) / _ \  | |\/| |/ _ \|  _/    
       \___|\___/|___/_/_\_\_|_|__|_/_/_\_\_|_ ___ 
            / __| | | | _ \ __| _ \___| _ \ __/ __|
            \__ \ |_| |  _/ _||   /___|   / _|\__ \
            |___/\___/|_| |___|_|_\   |_|_\___|___/
                                               2012
     
        by Jens Wetzl           (jens.wetzl@fau.de)
       and Oliver Taubmann (oliver.taubmann@fau.de)
     
       This work is licensed under a Creative Commons     
       Attribution 3.0 Unported License. (CC-BY)
       http://creativecommons.org/licenses/by/3.0/
      
===========================================================

This is a cross-platform, CUDA-based C++ implementation of 
the framework proposed in our paper "GPU Accelerated 
Time-of-Flight Super-Resolution for Image-Guided Surgery". 
It employs a maximum a posteriori (MAP) estimation to 
super-resolve an arbitrary, preregistered grayscale image 
sequence to obtain a single new image of improved quality 
and resolution. In particular, it can be used to enhance 
depth maps from range sensors such as Time-of-Flight 
cameras.

If you use this framework in your research, please cite:

Wetzl, J., Taubmann, O., Haase, S., Köhler, T., Kraus, M., 
and Hornegger, J. (2013). GPU-Accelerated Time-of-Flight 
Super-Resolution for Image-Guided Surgery. In Meinzer, 
H.-P., Deserno, T. M., Handels, H., and Tolxdorff, T., 
editors, Bildverarbeitung für die Medizin 2013, Informatik
aktuell, pages 21–26. Springer Berlin Heidelberg.

===========================================================
  DEPENDENCIES
===========================================================

To use this software, you need:

- CMake (http://www.cmake.org/) for generating build files 
  of your choice.

- The Nvidia GPU Computing Toolkit and SDK
  (http://www.nvidia.com/object/cuda_home_new.html).
  
- CUDA L-BFGS (https://github.com/jwetzl/CudaLBFGS), our 
  own library for GPU-accelerated nonlinear optimization.
  
- FreeImage (http://freeimage.sourceforge.net/), a 
  lightweight image IO library. Note: This can easily be 
  replaced with your preferred tool by adapting 
  ImageIO.{h,cpp} accordingly.

===========================================================
  BUILDING
===========================================================

The default settings should be fine for regular use, but 
there are some options, you can

- enable error checking and timing

- choose not to store the transpose of the system matrix.
  This will increase computation time but decrease the 
  memory footprint.

===========================================================
  USAGE
===========================================================

The superres binary displays a usage message when you run
it without parameters.
