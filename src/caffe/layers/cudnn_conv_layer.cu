#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  
  // Get cuDNN handle
  cudnnHandle_t handle = Caffe::cudnn_handle();
  
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    
    // Ensure workspace is large enough
    size_t workspace_needed = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        bottom_descs_[i]->Get(),
        filter_desc_->Get(),
        conv_descs_[i]->Get(),
        top_descs_[i]->Get(),
        fwd_algo_[i],
        &workspace_needed));
        
    if (workspace_needed > workspace_size_) {
      if (workspace_data_ != nullptr) {
        cudaFree(workspace_data_);
      }
      workspace_size_ = workspace_needed;
      cudaMalloc(&workspace_data_, workspace_size_);
    }
    
    // Forward pass
    CUDNN_CHECK(cudnnConvolutionForward(
        handle,
        cudnn::dataType<Dtype>::one,
        bottom_descs_[i]->Get(),
        bottom_data,
        filter_desc_->Get(),
        weight,
        conv_descs_[i]->Get(),
        fwd_algo_[i],
        workspace_data_,
        workspace_size_,
        cudnn::dataType<Dtype>::zero,
        top_descs_[i]->Get(),
        top_data));
        
    // Add bias
    if (this->bias_term_) {
      const Dtype* bias_data = this->blobs_[1]->gpu_data();
      CUDNN_CHECK(cudnnAddTensor(
          handle,
          cudnn::dataType<Dtype>::one,
          bias_desc_->Get(),
          bias_data,
          cudnn::dataType<Dtype>::one,
          top_descs_[i]->Get(),
          top_data));
    }
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  
  // Get cuDNN handle
  cudnnHandle_t handle = Caffe::cudnn_handle();
  
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    
    // Bias gradient
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      CUDNN_CHECK(cudnnConvolutionBackwardBias(
          handle,
          cudnn::dataType<Dtype>::one,
          top_descs_[i]->Get(),
          top_diff,
          cudnn::dataType<Dtype>::one,
          bias_desc_->Get(),
          bias_diff));
    }
    
    // Weight gradient
    if (this->param_propagate_down_[0]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      
      // Ensure workspace is large enough
      size_t workspace_needed = 0;
      CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
          handle,
          bottom_descs_[i]->Get(),
          top_descs_[i]->Get(),
          conv_descs_[i]->Get(),
          filter_desc_->Get(),
          bwd_filter_algo_[i],
          &workspace_needed));
          
      if (workspace_needed > workspace_size_) {
        if (workspace_data_ != nullptr) {
          cudaFree(workspace_data_);
        }
        workspace_size_ = workspace_needed;
        cudaMalloc(&workspace_data_, workspace_size_);
      }
      
      // Compute weight gradient
      CUDNN_CHECK(cudnnConvolutionBackwardFilter(
          handle,
          cudnn::dataType<Dtype>::one,
          bottom_descs_[i]->Get(),
          bottom_data,
          top_descs_[i]->Get(),
          top_diff,
          conv_descs_[i]->Get(),
          bwd_filter_algo_[i],
          workspace_data_,
          workspace_size_,
          cudnn::dataType<Dtype>::one,
          filter_desc_->Get(),
          weight_diff));
    }
    
    // Bottom diff
    if (propagate_down[i]) {
      // Ensure workspace is large enough
      size_t workspace_needed = 0;
      CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
          handle,
          filter_desc_->Get(),
          top_descs_[i]->Get(),
          conv_descs_[i]->Get(),
          bottom_descs_[i]->Get(),
          bwd_data_algo_[i],
          &workspace_needed));
          
      if (workspace_needed > workspace_size_) {
        if (workspace_data_ != nullptr) {
          cudaFree(workspace_data_);
        }
        workspace_size_ = workspace_needed;
        cudaMalloc(&workspace_data_, workspace_size_);
      }
      
      // Compute bottom diff
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      CUDNN_CHECK(cudnnConvolutionBackwardData(
          handle,
          cudnn::dataType<Dtype>::one,
          filter_desc_->Get(),
          weight,
          top_descs_[i]->Get(),
          top_diff,
          conv_descs_[i]->Get(),
          bwd_data_algo_[i],
          workspace_data_,
          workspace_size_,
          cudnn::dataType<Dtype>::zero,
          bottom_descs_[i]->Get(),
          bottom_diff));
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif