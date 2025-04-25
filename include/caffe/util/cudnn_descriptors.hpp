// include/caffe/util/cudnn_descriptors.hpp
#ifndef CAFFE_CUDNN_DESCRIPTORS_HPP_
#define CAFFE_CUDNN_DESCRIPTORS_HPP_

#include <cudnn.h>
#include <memory>
#include <vector>
#include "caffe/common.hpp"

namespace caffe {

class CuDNNTensorDescriptor {
public:
    CuDNNTensorDescriptor();
    ~CuDNNTensorDescriptor();

    void Set4D(int n, int c, int h, int w, cudnnDataType_t dtype = CUDNN_DATA_FLOAT);
    void SetNd(const std::vector<int>& dims, cudnnDataType_t dtype = CUDNN_DATA_FLOAT);
    cudnnTensorDescriptor_t Get() const { return descriptor_; }

private:
    cudnnTensorDescriptor_t descriptor_;
    DISABLE_COPY_AND_ASSIGN(CuDNNTensorDescriptor);
};

class CuDNNFilterDescriptor {
public:
    CuDNNFilterDescriptor();
    ~CuDNNFilterDescriptor();

    void Set4D(int n, int c, int h, int w, 
               cudnnDataType_t dtype = CUDNN_DATA_FLOAT,
               cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW);
    cudnnFilterDescriptor_t Get() const { return descriptor_; }

private:
    cudnnFilterDescriptor_t descriptor_;
    DISABLE_COPY_AND_ASSIGN(CuDNNFilterDescriptor);
};

class CuDNNConvolutionDescriptor {
public:
    CuDNNConvolutionDescriptor();
    ~CuDNNConvolutionDescriptor();

    void SetConv2D(int pad_h, int pad_w,
                   int stride_h, int stride_w,
                   int dilation_h = 1, int dilation_w = 1,
                   cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION,
                   cudnnDataType_t compute_type = CUDNN_DATA_FLOAT);
    void SetMathType(cudnnMathType_t math_type);
#if CUDNN_VERSION_MIN(9, 6, 0)
    void SetReorderType(cudnnReorderType_t reorder_type);
#endif
    cudnnConvolutionDescriptor_t Get() const { return descriptor_; }

private:
    cudnnConvolutionDescriptor_t descriptor_;
    DISABLE_COPY_AND_ASSIGN(CuDNNConvolutionDescriptor);
};

}  // namespace caffe

#endif  // CAFFE_CUDNN_DESCRIPTORS_HPP_