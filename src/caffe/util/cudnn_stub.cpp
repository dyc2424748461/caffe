#ifdef CPU_ONLY

#include "caffe/util/cudnn.h"
#include <cstring>

// Stub implementations for CPU_ONLY mode
cudnnStatus_t cudnnCreate(cudnnHandle_t* handle) {
    *handle = nullptr;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
    return CUDNN_STATUS_SUCCESS;
}

const char* cudnnGetErrorString(cudnnStatus_t status) {
    switch (status) {
        case CUDNN_STATUS_SUCCESS: return "CUDNN_STATUS_SUCCESS";
        case CUDNN_STATUS_NOT_INITIALIZED: return "CUDNN_STATUS_NOT_INITIALIZED";
        case CUDNN_STATUS_ALLOC_FAILED: return "CUDNN_STATUS_ALLOC_FAILED";
        case CUDNN_STATUS_BAD_PARAM: return "CUDNN_STATUS_BAD_PARAM";
        case CUDNN_STATUS_INTERNAL_ERROR: return "CUDNN_STATUS_INTERNAL_ERROR";
        case CUDNN_STATUS_INVALID_VALUE: return "CUDNN_STATUS_INVALID_VALUE";
        case CUDNN_STATUS_ARCH_MISMATCH: return "CUDNN_STATUS_ARCH_MISMATCH";
        case CUDNN_STATUS_MAPPING_ERROR: return "CUDNN_STATUS_MAPPING_ERROR";
        case CUDNN_STATUS_EXECUTION_FAILED: return "CUDNN_STATUS_EXECUTION_FAILED";
        case CUDNN_STATUS_NOT_SUPPORTED: return "CUDNN_STATUS_NOT_SUPPORTED";
        case CUDNN_STATUS_LICENSE_ERROR: return "CUDNN_STATUS_LICENSE_ERROR";
        case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING: return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
        case CUDNN_STATUS_RUNTIME_IN_PROGRESS: return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
        case CUDNN_STATUS_RUNTIME_FP_OVERFLOW: return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
        case CUDNN_STATUS_NOT_PERMITTED: return "CUDNN_STATUS_NOT_PERMITTED";
        case CUDNN_STATUS_INSUFFICIENT_DRIVER: return "CUDNN_STATUS_INSUFFICIENT_DRIVER";
        case CUDNN_STATUS_GRAPH_EXEC_ERROR: return "CUDNN_STATUS_GRAPH_EXEC_ERROR";
        default: return "Unknown cuDNN error";
    }
}

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc) {
    *tensorDesc = nullptr;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                        cudnnTensorFormat_t format,
                                        cudnnDataType_t dataType,
                                        int n, int c, int h, int w) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                        cudnnDataType_t dataType,
                                        int nbDims,
                                        const int dimA[],
                                        const int strideA[]) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* filterDesc) {
    *filterDesc = nullptr;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                        cudnnDataType_t dataType,
                                        cudnnTensorFormat_t format,
                                        int k, int c, int h, int w) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* convDesc) {
    *convDesc = nullptr;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                            int pad_h, int pad_w,
                                            int u, int v,
                                            int dilation_h, int dilation_w,
                                            cudnnConvolutionMode_t mode,
                                            cudnnDataType_t computeType) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                        cudnnMathType_t mathType) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                                           cudnnReorderType_t reorderType) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t srcDesc,
    cudnnFilterDescriptor_t filterDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t destDesc,
    int requestedAlgoCount,
    int* returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t* perfResults) {
    
    if (returnedAlgoCount) *returnedAlgoCount = 1;
    if (perfResults) {
        perfResults[0].algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        perfResults[0].status = CUDNN_STATUS_SUCCESS;
        perfResults[0].time = 0.0f;
        perfResults[0].memory = 0;
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t srcDesc,
    cudnnTensorDescriptor_t diffDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnFilterDescriptor_t gradDesc,
    int requestedAlgoCount,
    int* returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t* perfResults) {
    
    if (returnedAlgoCount) *returnedAlgoCount = 1;
    if (perfResults) {
        perfResults[0].algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        perfResults[0].status = CUDNN_STATUS_SUCCESS;
        perfResults[0].time = 0.0f;
        perfResults[0].memory = 0;
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle,
    cudnnFilterDescriptor_t filterDesc,
    cudnnTensorDescriptor_t diffDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t gradDesc,
    int requestedAlgoCount,
    int* returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t* perfResults) {
    
    if (returnedAlgoCount) *returnedAlgoCount = 1;
    if (perfResults) {
        perfResults[0].algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        perfResults[0].status = CUDNN_STATUS_SUCCESS;
        perfResults[0].time = 0.0f;
        perfResults[0].memory = 0;
    }
    return CUDNN_STATUS_SUCCESS;
}

#endif // CPU_ONLY