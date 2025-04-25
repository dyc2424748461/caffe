// src/caffe/util/cudnn_wrapper.cpp
#include <sstream>
#include <stdexcept>
#include "caffe/util/cudnn_wrapper.hpp"

namespace caffe {

CuDNNWrapper& CuDNNWrapper::GetInstance() {
    static CuDNNWrapper instance;
    return instance;
}

CuDNNWrapper::CuDNNWrapper() {
    CUDNN_CHECK(cudnnCreate(&handle_));
}

CuDNNWrapper::~CuDNNWrapper() {
    if (handle_) {
        cudnnDestroy(handle_);
    }
}

cudnnHandle_t CuDNNWrapper::GetHandle() const {
    return handle_;
}

bool CuDNNWrapper::IsVersion900OrHigher() const {
    return CUDNN_VERSION >= 9000;
}

bool CuDNNWrapper::IsVersion960OrHigher() const {
    return CUDNN_VERSION >= 9600;
}

std::string CuDNNWrapper::GetVersionString() const {
    int version = CUDNN_VERSION;
    std::stringstream ss;
    ss << (version / 1000) << "." 
       << (version % 1000 / 100) << "."
       << (version % 100);
    return ss.str();
}

void CuDNNWrapper::CheckError(cudnnStatus_t status, const char* file, int line) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("CUDNN error at ") + file + ":" + 
            std::to_string(line) + ": " + 
            cudnnGetErrorString(status));
    }
}

} // namespace caffe