// src/caffe/util/cudnn_algorithm_selector.cpp
#include "caffe/util/cudnn_algorithm_selector.hpp"
#include <algorithm>

namespace caffe {

CuDNNAlgorithmSelector::CuDNNAlgorithmSelector() {}

cudnnConvolutionFwdAlgo_t CuDNNAlgorithmSelector::SelectForwardAlgorithm(
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    size_t workspace_limit) {
    
    int algo_count;
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
    
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        CuDNNWrapper::GetInstance().GetHandle(),
        xDesc,
        wDesc,
        convDesc,
        yDesc,
        perf_results.size(),
        &algo_count,
        perf_results.data()));
    
    perf_results.resize(algo_count);
    return perf_results[SelectBestAlgorithm(perf_results, workspace_limit)].algo;
}

cudnnConvolutionBwdFilterAlgo_t CuDNNAlgorithmSelector::SelectBackwardFilterAlgorithm(
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    size_t workspace_limit) {
    
    int algo_count;
    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT);
    
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        CuDNNWrapper::GetInstance().GetHandle(),
        xDesc,
        dyDesc,
        convDesc,
        dwDesc,
        perf_results.size(),
        &algo_count,
        perf_results.data()));
    
    perf_results.resize(algo_count);
    return perf_results[SelectBestAlgorithm(perf_results, workspace_limit)].algo;
}

cudnnConvolutionBwdDataAlgo_t CuDNNAlgorithmSelector::SelectBackwardDataAlgorithm(
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    size_t workspace_limit) {
    
    int algo_count;
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT);
    
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        CuDNNWrapper::GetInstance().GetHandle(),
        wDesc,
        dyDesc,
        convDesc,
        dxDesc,
        perf_results.size(),
        &algo_count,
        perf_results.data()));
    
    perf_results.resize(algo_count);
    return perf_results[SelectBestAlgorithm(perf_results, workspace_limit)].algo;
}

template<typename AlgoPerf_t>
int CuDNNAlgorithmSelector::SelectBestAlgorithm(
    const std::vector<AlgoPerf_t>& perf_results,
    size_t workspace_limit) {
    
    for (int i = 0; i < perf_results.size(); ++i) {
        if (perf_results[i].status == CUDNN_STATUS_SUCCESS &&
            perf_results[i].memory <= workspace_limit) {
            return i;
        }
    }
    
    // If no algorithm fits in workspace_limit, choose the one with minimum memory
    return std::min_element(
        perf_results.begin(),
        perf_results.end(),
        [](const AlgoPerf_t& a, const AlgoPerf_t& b) {
            return a.memory < b.memory;
        }) - perf_results.begin();
}

} // namespace caffe