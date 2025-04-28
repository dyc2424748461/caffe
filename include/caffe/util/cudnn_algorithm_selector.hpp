#ifndef CAFFE_UTIL_CUDNN_ALGORITHM_SELECTOR_HPP_
#define CAFFE_UTIL_CUDNN_ALGORITHM_SELECTOR_HPP_

#ifdef USE_CUDNN

#include <cudnn.h>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/util/cudnn.hpp"

namespace caffe {

// Class to handle algorithm selection for cuDNN operations
class CuDNNAlgorithmSelector {
 public:
  CuDNNAlgorithmSelector() {
    CUDNN_CHECK(cudnnCreate(&handle_));
  }

  ~CuDNNAlgorithmSelector() {
    CUDNN_CHECK(cudnnDestroy(handle_));
  }

  // Select forward convolution algorithm
  cudnnConvolutionFwdAlgo_t SelectForwardAlgorithm(
      cudnnTensorDescriptor_t bottom_desc,
      cudnnFilterDescriptor_t filter_desc,
      cudnnConvolutionDescriptor_t conv_desc,
      cudnnTensorDescriptor_t top_desc,
      size_t workspace_limit_bytes) {
    
    int num_algos;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithmMaxCount(handle_, &num_algos));
    
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(num_algos);
    int returned_algos;
    
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
        handle_,
        bottom_desc,
        filter_desc,
        conv_desc,
        top_desc,
        num_algos,
        &returned_algos,
        perf_results.data()));
    
    // Find the fastest algorithm that fits within our memory limit
    for (int i = 0; i < returned_algos; i++) {
      if (perf_results[i].status == CUDNN_STATUS_SUCCESS &&
          perf_results[i].memory <= workspace_limit_bytes) {
        return perf_results[i].algo;
      }
    }
    
    // If no algorithm fits, use the no-workspace algorithm
    return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  }

  // Select backward filter convolution algorithm
  cudnnConvolutionBwdFilterAlgo_t SelectBackwardFilterAlgorithm(
      cudnnTensorDescriptor_t bottom_desc,
      cudnnTensorDescriptor_t top_desc,
      cudnnConvolutionDescriptor_t conv_desc,
      cudnnFilterDescriptor_t filter_desc,
      size_t workspace_limit_bytes) {
    
    int num_algos;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle_, &num_algos));
    
    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(num_algos);
    int returned_algos;
    
    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(
        handle_,
        bottom_desc,
        top_desc,
        conv_desc,
        filter_desc,
        num_algos,
        &returned_algos,
        perf_results.data()));
    
    // Find the fastest algorithm that fits within our memory limit
    for (int i = 0; i < returned_algos; i++) {
      if (perf_results[i].status == CUDNN_STATUS_SUCCESS &&
          perf_results[i].memory <= workspace_limit_bytes) {
        return perf_results[i].algo;
      }
    }
    
    // If no algorithm fits, use the no-workspace algorithm
    return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  }

  // Select backward data convolution algorithm
  cudnnConvolutionBwdDataAlgo_t SelectBackwardDataAlgorithm(
      cudnnFilterDescriptor_t filter_desc,
      cudnnTensorDescriptor_t top_desc,
      cudnnConvolutionDescriptor_t conv_desc,
      cudnnTensorDescriptor_t bottom_desc,
      size_t workspace_limit_bytes) {
    
    int num_algos;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle_, &num_algos));
    
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(num_algos);
    int returned_algos;
    
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
        handle_,
        filter_desc,
        top_desc,
        conv_desc,
        bottom_desc,
        num_algos,
        &returned_algos,
        perf_results.data()));
    
    // Find the fastest algorithm that fits within our memory limit
    for (int i = 0; i < returned_algos; i++) {
      if (perf_results[i].status == CUDNN_STATUS_SUCCESS &&
          perf_results[i].memory <= workspace_limit_bytes) {
        return perf_results[i].algo;
      }
    }
    
    // If no algorithm fits, use the no-workspace algorithm
    return CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  }

 private:
  cudnnHandle_t handle_;
};

}  // namespace caffe

#endif  // USE_CUDNN
#endif  // CAFFE_UTIL_CUDNN_ALGORITHM_SELECTOR_HPP_