// Mock cudnn.h for CPU_ONLY mode
#ifndef CAFFE_UTIL_CUDNN_H_
#define CAFFE_UTIL_CUDNN_H_

#include <cstddef>

// Define cuDNN version 9.6.0
#define CUDNN_VERSION 9600

// Define CUDNN_VERSION_MIN macro
#ifndef CUDNN_VERSION_MIN
#define CUDNN_VERSION_MIN(major, minor, patch) \
    (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))
#endif

// Define cuDNN status codes
typedef enum {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1,
    CUDNN_STATUS_ALLOC_FAILED = 2,
    CUDNN_STATUS_BAD_PARAM = 3,
    CUDNN_STATUS_INTERNAL_ERROR = 4,
    CUDNN_STATUS_INVALID_VALUE = 5,
    CUDNN_STATUS_ARCH_MISMATCH = 6,
    CUDNN_STATUS_MAPPING_ERROR = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
    CUDNN_STATUS_LICENSE_ERROR = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13,
    CUDNN_STATUS_NOT_PERMITTED = 14,
    CUDNN_STATUS_INSUFFICIENT_DRIVER = 15,
    CUDNN_STATUS_GRAPH_EXEC_ERROR = 16
} cudnnStatus_t;

// Define cuDNN data types
typedef enum {
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_DOUBLE = 1
} cudnnDataType_t;

// Define cuDNN tensor format
typedef enum {
    CUDNN_TENSOR_NCHW = 0
} cudnnTensorFormat_t;

// Define cuDNN convolution mode
typedef enum {
    CUDNN_CROSS_CORRELATION = 0
} cudnnConvolutionMode_t;

// Define cuDNN math type
typedef enum {
    CUDNN_DEFAULT_MATH = 0,
    CUDNN_TENSOR_OP_MATH = 1
} cudnnMathType_t;

// Define cuDNN reorder type
typedef enum {
    CUDNN_DEFAULT_REORDER = 0
} cudnnReorderType_t;

// Define cuDNN pooling mode
typedef enum {
    CUDNN_POOLING_MAX = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1
} cudnnPoolingMode_t;

// Define cuDNN propagate NaN
typedef enum {
    CUDNN_PROPAGATE_NAN = 0
} cudnnNanPropagation_t;

// Define cuDNN activation mode
typedef enum {
    CUDNN_ACTIVATION_SIGMOID = 0,
    CUDNN_ACTIVATION_RELU = 1,
    CUDNN_ACTIVATION_TANH = 2
} cudnnActivationMode_t;

// Define cuDNN convolution algorithms
typedef enum {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0
} cudnnConvolutionFwdAlgo_t;

typedef enum {
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0
} cudnnConvolutionBwdFilterAlgo_t;

typedef enum {
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0
} cudnnConvolutionBwdDataAlgo_t;

// Define cuDNN algorithm performance structures
typedef struct {
    cudnnConvolutionFwdAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
} cudnnConvolutionFwdAlgoPerf_t;

typedef struct {
    cudnnConvolutionBwdFilterAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
} cudnnConvolutionBwdFilterAlgoPerf_t;

typedef struct {
    cudnnConvolutionBwdDataAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
} cudnnConvolutionBwdDataAlgoPerf_t;

// Define cuDNN handle and descriptor types
typedef void* cudnnHandle_t;
typedef void* cudnnTensorDescriptor_t;
typedef void* cudnnFilterDescriptor_t;
typedef void* cudnnConvolutionDescriptor_t;
typedef void* cudnnPoolingDescriptor_t;
typedef void* cudnnActivationDescriptor_t;

// Define constants
#define CUDNN_CONVOLUTION_FWD_ALGO_COUNT 8
#define CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT 8
#define CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT 8

// Function declarations for CPU_ONLY mode (stubs)
cudnnStatus_t cudnnCreate(cudnnHandle_t* handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);
const char* cudnnGetErrorString(cudnnStatus_t status);

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                        cudnnTensorFormat_t format,
                                        cudnnDataType_t dataType,
                                        int n, int c, int h, int w);
cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                        cudnnDataType_t dataType,
                                        int nbDims,
                                        const int dimA[],
                                        const int strideA[]);

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* filterDesc);
cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc);
cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                        cudnnDataType_t dataType,
                                        cudnnTensorFormat_t format,
                                        int k, int c, int h, int w);

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* convDesc);
cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc);
cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                            int pad_h, int pad_w,
                                            int u, int v,
                                            int dilation_h, int dilation_w,
                                            cudnnConvolutionMode_t mode,
                                            cudnnDataType_t computeType);

cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                        cudnnMathType_t mathType);
                                        
cudnnStatus_t cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                                           cudnnReorderType_t reorderType);

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t srcDesc,
    cudnnFilterDescriptor_t filterDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t destDesc,
    int requestedAlgoCount,
    int* returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t* perfResults);

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t srcDesc,
    cudnnTensorDescriptor_t diffDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnFilterDescriptor_t gradDesc,
    int requestedAlgoCount,
    int* returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t* perfResults);

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle,
    cudnnFilterDescriptor_t filterDesc,
    cudnnTensorDescriptor_t diffDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t gradDesc,
    int requestedAlgoCount,
    int* returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t* perfResults);

#endif // CAFFE_UTIL_CUDNN_H_