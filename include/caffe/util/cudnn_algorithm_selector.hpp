// include/caffe/util/cudnn_algorithm_selector.hpp
#ifndef CAFFE_CUDNN_ALGORITHM_SELECTOR_HPP_
#define CAFFE_CUDNN_ALGORITHM_SELECTOR_HPP_

#include <cudnn.h>
#include <vector>
#include "caffe/util/cudnn_wrapper.hpp"

namespace caffe {

class CuDNNAlgorithmSelector {
public:
    CuDNNAlgorithmSelector();

    cudnnConvolutionFwdAlgo_t SelectForwardAlgorithm(
        const cudnnTensorDescriptor_t xDesc,
        const cudnnFilterDescriptor_t wDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t yDesc,
        size_t workspace_limit);

    cudnnConvolutionBwdFilterAlgo_t SelectBackwardFilterAlgorithm(
        const cudnnTensorDescriptor_t xDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnFilterDescriptor_t dwDesc,
        size_t workspace_limit);

    cudnnConvolutionBwdDataAlgo_t SelectBackwardDataAlgorithm(
        const cudnnFilterDescriptor_t wDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t dxDesc,
        size_t workspace_limit);

private:
    template<typename AlgoPerf_t>
    static int SelectBestAlgorithm(
        const std::vector<AlgoPerf_t>& perf_results,
        size_t workspace_limit);
};

} // namespace caffe

#endif // CAFFE_CUDNN_ALGORITHM_SELECTOR_HPP_