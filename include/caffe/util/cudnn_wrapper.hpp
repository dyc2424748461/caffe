// include/caffe/util/cudnn_wrapper.hpp
#ifndef CAFFE_CUDNN_WRAPPER_HPP_
#define CAFFE_CUDNN_WRAPPER_HPP_

#include <cudnn.h>
#include <memory>
#include <string>

namespace caffe {

class CuDNNWrapper {
public:
    static CuDNNWrapper& GetInstance();
    cudnnHandle_t GetHandle() const;
    
    // Version check utilities
    bool IsVersion900OrHigher() const;
    bool IsVersion960OrHigher() const;
    std::string GetVersionString() const;

    // Error handling
    static void CheckError(cudnnStatus_t status, const char* file, int line);

private:
    CuDNNWrapper();
    ~CuDNNWrapper();
    
    // Prevent copying
    CuDNNWrapper(const CuDNNWrapper&) = delete;
    CuDNNWrapper& operator=(const CuDNNWrapper&) = delete;

    cudnnHandle_t handle_;
};

// Macro for error checking
#define CUDNN_CHECK(condition) \
    CuDNNWrapper::CheckError((condition), __FILE__, __LINE__)

} // namespace caffe

#endif // CAFFE_CUDNN_WRAPPER_HPP_