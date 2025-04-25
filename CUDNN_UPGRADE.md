# cuDNN 9.6.0 Upgrade for Caffe

This document describes the changes made to adapt Caffe to work with cuDNN 9.6.0.

## Overview of Changes

1. Added support for cuDNN 9.6.0 API in the following files:
   - `include/caffe/util/cudnn.hpp`: Added new error codes for cuDNN 9.6.0
   - `include/caffe/util/cudnn_wrapper.hpp`: Added version check methods
   - `src/caffe/util/cudnn_wrapper.cpp`: Implemented version check methods
   - `include/caffe/util/cudnn_descriptors.hpp`: Added SetReorderType method for cuDNN 9.6.0
   - `src/caffe/util/cudnn_descriptors.cpp`: Updated filter descriptor and convolution descriptor APIs
   - `src/caffe/layers/cudnn_conv_layer.cpp`: Added support for cuDNN 9.6.0 specific features

2. Added CPU_ONLY mode support:
   - `include/caffe/util/cudnn.h`: Mock header for cuDNN when building in CPU_ONLY mode
   - `src/caffe/util/cudnn_stub.cpp`: Stub implementations for cuDNN functions in CPU_ONLY mode

3. Fixed compatibility issues:
   - Updated OpenCV API calls in `src/caffe/util/io.cpp` (replaced CV_LOAD_IMAGE_COLOR with cv::IMREAD_COLOR)
   - Updated protobuf API in `src/caffe/util/io.cpp` (updated SetTotalBytesLimit call)

## API Changes in cuDNN 9.6.0

1. New error codes:
   - `CUDNN_STATUS_NOT_PERMITTED`
   - `CUDNN_STATUS_INSUFFICIENT_DRIVER`
   - `CUDNN_STATUS_GRAPH_EXEC_ERROR`

2. New features:
   - Added support for `cudnnReorderType_t` and `cudnnSetConvolutionReorderType`
   - Updated filter descriptor API

## Building with cuDNN 9.6.0

To build with cuDNN 9.6.0 support:

1. Install cuDNN 9.6.0 SDK
2. Set the CUDNN_ROOT environment variable to point to your cuDNN installation
3. Build Caffe normally:
   ```
   mkdir build && cd build
   cmake .. -DUSE_CUDNN=ON
   make -j4
   ```

To build in CPU_ONLY mode (no CUDA/cuDNN required):
```
mkdir build && cd build
cmake .. -DCPU_ONLY=ON
make -j4
```

## Compatibility

This version of Caffe is compatible with:
- cuDNN 9.6.0
- cuDNN 9.0.0 (with reduced functionality)
- CPU-only mode (no CUDA/cuDNN required)

The code includes conditional compilation to maintain backward compatibility with older cuDNN versions where possible.