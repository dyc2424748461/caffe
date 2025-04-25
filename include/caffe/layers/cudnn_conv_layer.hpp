// include/caffe/layers/cudnn_conv_layer.hpp
#ifndef CAFFE_CUDNN_CONV_LAYER_HPP_
#define CAFFE_CUDNN_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/cudnn_wrapper.hpp"
#include "caffe/util/cudnn_descriptors.hpp"
#include "caffe/util/cudnn_algorithm_selector.hpp"

namespace caffe {

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNConvolutionLayer : public ConvolutionLayer<Dtype> {
 public:
  explicit CuDNNConvolutionLayer(const LayerParameter& param)
      : ConvolutionLayer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNConvolutionLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool handles_setup_;
  
  // Algorithms for forward and backward passes
  vector<cudnnConvolutionFwdAlgo_t> fwd_algo_;
  vector<cudnnConvolutionBwdFilterAlgo_t> bwd_filter_algo_;
  vector<cudnnConvolutionBwdDataAlgo_t> bwd_data_algo_;

  // Descriptors
  vector<shared_ptr<CuDNNTensorDescriptor>> bottom_descs_;
  vector<shared_ptr<CuDNNTensorDescriptor>> top_descs_;
  shared_ptr<CuDNNTensorDescriptor> bias_desc_;
  shared_ptr<CuDNNFilterDescriptor> filter_desc_;
  vector<shared_ptr<CuDNNConvolutionDescriptor>> conv_descs_;

  // Algorithm selector
  shared_ptr<CuDNNAlgorithmSelector> algorithm_selector_;

  // Workspace management
  size_t workspace_size_;
  void* workspace_data_;
  
  int bottom_offset_, top_offset_, bias_offset_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_CONV_LAYER_HPP_
