#ifndef CAFFE_UTIL_CUDNN_DESCRIPTORS_HPP_
#define CAFFE_UTIL_CUDNN_DESCRIPTORS_HPP_

#ifdef USE_CUDNN

#include <cudnn.h>
#include <memory>
#include "caffe/common.hpp"
#include "caffe/util/cudnn.hpp"

namespace caffe {

// Base class for all cuDNN descriptor wrappers
class CuDNNDescriptorBase {
 public:
  virtual ~CuDNNDescriptorBase() {}
};

// Tensor descriptor wrapper
class CuDNNTensorDescriptor : public CuDNNDescriptorBase {
 public:
  CuDNNTensorDescriptor() {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
  }

  ~CuDNNTensorDescriptor() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc_));
  }

  void Set4D(int n, int c, int h, int w) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc_,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));
  }

  cudnnTensorDescriptor_t Get() const {
    return desc_;
  }

 private:
  cudnnTensorDescriptor_t desc_;
};

// Filter descriptor wrapper
class CuDNNFilterDescriptor : public CuDNNDescriptorBase {
 public:
  CuDNNFilterDescriptor() {
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc_));
  }

  ~CuDNNFilterDescriptor() {
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(desc_));
  }

#if CUDNN_VERSION_MIN(9, 0, 0)
  void Set4D(int n, int c, int h, int w, cudnnDataType_t dataType, cudnnTensorFormat_t format) {
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(desc_, dataType, format, n, c, h, w));
  }
#else
  void Set4D(int n, int c, int h, int w) {
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w));
  }
#endif

  cudnnFilterDescriptor_t Get() const {
    return desc_;
  }

 private:
  cudnnFilterDescriptor_t desc_;
};

// Convolution descriptor wrapper
class CuDNNConvolutionDescriptor : public CuDNNDescriptorBase {
 public:
  CuDNNConvolutionDescriptor() {
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc_));
  }

  ~CuDNNConvolutionDescriptor() {
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(desc_));
  }

#if CUDNN_VERSION_MIN(9, 0, 0)
  void SetConv2D(int pad_h, int pad_w, int stride_h, int stride_w, 
                int dilation_h, int dilation_w, cudnnConvolutionMode_t mode, 
                cudnnDataType_t dataType) {
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(desc_, 
                                              pad_h, pad_w, 
                                              stride_h, stride_w, 
                                              dilation_h, dilation_w, 
                                              mode, dataType));
  }

  void SetMathType(cudnnMathType_t mathType) {
    CUDNN_CHECK(cudnnSetConvolutionMathType(desc_, mathType));
  }

#if CUDNN_VERSION_MIN(9, 6, 0)
  void SetReorderType(cudnnReorderType_t reorderType) {
    CUDNN_CHECK(cudnnSetConvolutionReorderType(desc_, reorderType));
  }
#endif

#else
  void SetConv2D(int pad_h, int pad_w, int stride_h, int stride_w) {
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(desc_, 
                                              pad_h, pad_w, 
                                              stride_h, stride_w, 
                                              1, 1, 
                                              CUDNN_CROSS_CORRELATION));
  }
#endif

  cudnnConvolutionDescriptor_t Get() const {
    return desc_;
  }

 private:
  cudnnConvolutionDescriptor_t desc_;
};

// Pooling descriptor wrapper
class CuDNNPoolingDescriptor : public CuDNNDescriptorBase {
 public:
  CuDNNPoolingDescriptor() {
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&desc_));
  }

  ~CuDNNPoolingDescriptor() {
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(desc_));
  }

  void Set2D(cudnnPoolingMode_t mode, int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(desc_, 
                                          mode, 
                                          CUDNN_PROPAGATE_NAN, 
                                          h, w, 
                                          pad_h, pad_w, 
                                          stride_h, stride_w));
  }

  cudnnPoolingDescriptor_t Get() const {
    return desc_;
  }

 private:
  cudnnPoolingDescriptor_t desc_;
};

// Activation descriptor wrapper
class CuDNNActivationDescriptor : public CuDNNDescriptorBase {
 public:
  CuDNNActivationDescriptor() {
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&desc_));
  }

  ~CuDNNActivationDescriptor() {
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(desc_));
  }

  void Set(cudnnActivationMode_t mode, double reluCoef) {
    CUDNN_CHECK(cudnnSetActivationDescriptor(desc_, mode, CUDNN_PROPAGATE_NAN, reluCoef));
  }

  cudnnActivationDescriptor_t Get() const {
    return desc_;
  }

 private:
  cudnnActivationDescriptor_t desc_;
};

}  // namespace caffe

#endif  // USE_CUDNN
#endif  // CAFFE_UTIL_CUDNN_DESCRIPTORS_HPP_