// src/caffe/layers/cudnn_conv_layer.cpp
#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  // Initialize algorithm selector
  algorithm_selector_.reset(new CuDNNAlgorithmSelector());

  // Initialize descriptors for each bottom blob
  for (int i = 0; i < bottom.size(); i++) {
    bottom_descs_.push_back(make_shared<CuDNNTensorDescriptor>());
    top_descs_.push_back(make_shared<CuDNNTensorDescriptor>());
    conv_descs_.push_back(make_shared<CuDNNConvolutionDescriptor>());
  }

  // Initialize filter descriptor
  filter_desc_.reset(new CuDNNFilterDescriptor());
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];

#if CUDNN_VERSION_MIN(9, 6, 0)
  filter_desc_->Set4D(
      this->num_output_ / this->group_,
      this->channels_ / this->group_,
      kernel_h, kernel_w,
      CUDNN_DATA_FLOAT,
      CUDNN_TENSOR_NCHW);
#elif CUDNN_VERSION_MIN(9, 0, 0)
  filter_desc_->Set4D(
      this->num_output_ / this->group_,
      this->channels_ / this->group_,
      kernel_h, kernel_w,
      CUDNN_DATA_FLOAT,
      CUDNN_TENSOR_NCHW);
#else
  filter_desc_->Set4D(
      this->num_output_ / this->group_,
      this->channels_ / this->group_,
      kernel_h, kernel_w);
#endif

  // Tensor descriptor for bias
  if (this->bias_term_) {
    bias_desc_.reset(new CuDNNTensorDescriptor());
  }

  // Initialize algorithm choices
  fwd_algo_.resize(bottom.size());
  bwd_filter_algo_.resize(bottom.size());
  bwd_data_algo_.resize(bottom.size());

  workspace_size_ = 0;
  workspace_data_ = nullptr;
  
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);

  CHECK_EQ(2, this->num_spatial_axes_)
      << "CuDNNConvolution input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND convolution.";

  bottom_offset_ = this->bottom_dim_ / this->group_;
  top_offset_ = this->top_dim_ / this->group_;
  
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  // Set tensor descriptors
  for (int i = 0; i < bottom.size(); i++) {
    bottom_descs_[i]->Set4D(
        this->num_,
        this->channels_ / this->group_,
        height, width);

    top_descs_[i]->Set4D(
        this->num_,
        this->num_output_ / this->group_,
        height_out, width_out);

#if CUDNN_VERSION_MIN(9, 6, 0)
    conv_descs_[i]->SetConv2D(
        pad_h, pad_w,
        stride_h, stride_w,
        1, 1,  // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT);
    // Enable tensor cores if available
    conv_descs_[i]->SetMathType(CUDNN_TENSOR_OP_MATH);
    // Set reorder type for cuDNN 9.6.0
    conv_descs_[i]->SetReorderType(CUDNN_DEFAULT_REORDER);
#elif CUDNN_VERSION_MIN(9, 0, 0)
    conv_descs_[i]->SetConv2D(
        pad_h, pad_w,
        stride_h, stride_w,
        1, 1,  // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT);
    // Enable tensor cores if available
    conv_descs_[i]->SetMathType(CUDNN_TENSOR_OP_MATH);
#else
    conv_descs_[i]->SetConv2D(
        pad_h, pad_w,
        stride_h, stride_w);
#endif

    // Select algorithms using the new algorithm selector
    size_t workspace_limit = 8 * 1024 * 1024;  // 8MB limit
    
    fwd_algo_[i] = algorithm_selector_->SelectForwardAlgorithm(
        bottom_descs_[i]->Get(),
        filter_desc_->Get(),
        conv_descs_[i]->Get(),
        top_descs_[i]->Get(),
        workspace_limit);

    bwd_filter_algo_[i] = algorithm_selector_->SelectBackwardFilterAlgorithm(
        bottom_descs_[i]->Get(),
        top_descs_[i]->Get(),
        conv_descs_[i]->Get(),
        filter_desc_->Get(),
        workspace_limit);

    bwd_data_algo_[i] = algorithm_selector_->SelectBackwardDataAlgorithm(
        filter_desc_->Get(),
        top_descs_[i]->Get(),
        conv_descs_[i]->Get(),
        bottom_descs_[i]->Get(),
        workspace_limit);
  }

  // Set bias descriptor
  if (this->bias_term_) {
    bias_desc_->Set4D(1, this->num_output_ / this->group_, 1, 1);
  }
}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  if (!handles_setup_) return;
  
  if (workspace_data_ != nullptr) {
    cudaFree(workspace_data_);
    workspace_data_ = nullptr;
  }
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
