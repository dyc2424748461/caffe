// src/caffe/util/cudnn_descriptors.cpp
#include "caffe/util/cudnn_descriptors.hpp"
#include "caffe/util/cudnn_wrapper.hpp"

namespace caffe {

// CuDNNTensorDescriptor implementation
CuDNNTensorDescriptor::CuDNNTensorDescriptor() {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&descriptor_));
}

CuDNNTensorDescriptor::~CuDNNTensorDescriptor() {
    if (descriptor_) {
        cudnnDestroyTensorDescriptor(descriptor_);
    }
}

void CuDNNTensorDescriptor::Set4D(int n, int c, int h, int w, cudnnDataType_t dtype) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(descriptor_,
        CUDNN_TENSOR_NCHW, dtype, n, c, h, w));
}

void CuDNNTensorDescriptor::SetNd(const std::vector<int>& dims, cudnnDataType_t dtype) {
    std::vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(descriptor_,
        dtype, dims.size(), dims.data(), strides.data()));
}

// CuDNNFilterDescriptor implementation
CuDNNFilterDescriptor::CuDNNFilterDescriptor() {
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&descriptor_));
}

CuDNNFilterDescriptor::~CuDNNFilterDescriptor() {
    if (descriptor_) {
        cudnnDestroyFilterDescriptor(descriptor_);
    }
}

void CuDNNFilterDescriptor::Set4D(int n, int c, int h, int w, 
                                 cudnnDataType_t dtype,
                                 cudnnTensorFormat_t format) {
#if CUDNN_VERSION_MIN(9, 6, 0)
    // Use newer API for cuDNN 9.6.0 and above
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(descriptor_,
        dtype, format, n, c, h, w));
#elif CUDNN_VERSION_MIN(9, 0, 0)
    // Use API for cuDNN 9.0.0 to 9.5.x
    CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(descriptor_,
        dtype, format, n, c, h, w));
#else
    // Use older API for earlier versions
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(descriptor_,
        dtype, format, n, c, h, w));
#endif
}

// CuDNNConvolutionDescriptor implementation
CuDNNConvolutionDescriptor::CuDNNConvolutionDescriptor() {
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&descriptor_));
}

CuDNNConvolutionDescriptor::~CuDNNConvolutionDescriptor() {
    if (descriptor_) {
        cudnnDestroyConvolutionDescriptor(descriptor_);
    }
}

void CuDNNConvolutionDescriptor::SetConv2D(
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    cudnnConvolutionMode_t mode,
    cudnnDataType_t compute_type) {
    
#if CUDNN_VERSION_MIN(9, 6, 0)
    // Use newer API with compute precision for cuDNN 9.6.0
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(descriptor_,
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, mode, compute_type));
#elif CUDNN_VERSION_MIN(9, 0, 0)
    // Use API for cuDNN 9.0.0 to 9.5.x
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(descriptor_,
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, mode, compute_type));
#else
    // Use older API for earlier versions
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(descriptor_,
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, mode));
#endif
}

void CuDNNConvolutionDescriptor::SetMathType(cudnnMathType_t math_type) {
#if CUDNN_VERSION_MIN(9, 0, 0)
    CUDNN_CHECK(cudnnSetConvolutionMathType(descriptor_, math_type));
#endif
}

#if CUDNN_VERSION_MIN(9, 6, 0)
void CuDNNConvolutionDescriptor::SetReorderType(cudnnReorderType_t reorder_type) {
    CUDNN_CHECK(cudnnSetConvolutionReorderType(descriptor_, reorder_type));
}
#endif

}  // namespace caffe