#ifndef CAFFE_UTIL_CUDNN_WRAPPER_HPP_
#define CAFFE_UTIL_CUDNN_WRAPPER_HPP_

#ifdef USE_CUDNN

#include <cudnn.h>
#include <memory>
#include "caffe/common.hpp"
#include "caffe/util/cudnn.hpp"

namespace caffe {

// cuDNN handle wrapper
class CuDNNHandle {
 public:
  CuDNNHandle() {
    CUDNN_CHECK(cudnnCreate(&handle_));
  }

  ~CuDNNHandle() {
    CUDNN_CHECK(cudnnDestroy(handle_));
  }

  cudnnHandle_t Get() const {
    return handle_;
  }

 private:
  cudnnHandle_t handle_;
};

}  // namespace caffe

#endif  // USE_CUDNN
#endif  // CAFFE_UTIL_CUDNN_WRAPPER_HPP_