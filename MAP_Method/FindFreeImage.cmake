find_path(FreeImage_INCLUDE_DIR FreeImage.h HINTS /usr/include /usr/local/include /opt/local/include ${FreeImage_DIR}/include)
find_library(FreeImage_LIBRARY NAMES freeimage HINTS /usr/lib /usr/local/lib /opt/local/lib ${FreeImage_DIR}/lib)

set(FreeImage_INCLUDE_DIRS ${FreeImage_INCLUDE_DIR})
set(FreeImage_LIBRARIES ${FreeImage_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FreeImage DEFAULT_MSG FreeImage_INCLUDE_DIR FreeImage_LIBRARY)

mark_as_advanced(FreeImage_INCLUDE_DIR FreeImage_LIBRARY)
