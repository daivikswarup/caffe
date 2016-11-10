#include <cfloat>
#include <vector>

#include "caffe/layers/dotprod_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DotProdLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // CHECK(this->layer_param().eltwise_param().coeff_size() == 0
  //     || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
  //     "Eltwise Layer takes one coefficient per bottom blob.";
  // CHECK(!(this->layer_param().eltwise_param().operation()
  //     == EltwiseParameter_EltwiseOp_PROD
  //     && this->layer_param().eltwise_param().coeff_size())) <<
  //     "Eltwise layer only takes coefficients for summation.";
  // op_ = this->layer_param_.eltwise_param().operation();
  // // Blob-wise coefficients for the elementwise operation.
  // coeffs_ = vector<Dtype>(bottom.size(), 1);
  // if (this->layer_param().eltwise_param().coeff_size()) {
  //   for (int i = 0; i < bottom.size(); ++i) {
  //     coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
  //   }
  // }
  // stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();
}

template <typename Dtype>
void DotProdLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // for (int i = 1; i < bottom.size(); ++i) {
  //   CHECK(bottom[i]->shape() == bottom[0]->shape());
  // }


  // Pliss ensure that blobs are appropriately sized. Will add the checks later. Pliss.
  top[0]->ReshapeLike(*bottom[0]);
  // If max operation, we will initialize the vector index part.

}

template <typename Dtype>
void DotProdLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const vector<int> blobshape = bottom[0]->shape();
  const int count = blobshape[2]*blobshape[3];//Number of pixels
  const int num_classes_batches = blobshape[0]*blobshape[1];
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weights = bottom[1]->cpu_data();
  for(int i=0;i<num_classes_batches;i++)
  {
      caffe_mul(count, bottom_data, weights, top_data);
      top_data = top_data + count;
      bottom_data = bottom_data + count;
  }
}

template <typename Dtype>
void DotProdLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const vector<int> blobshape = bottom[0]->shape();
  const int count = blobshape[2]*blobshape[3];//Number of pixels
  const int num_classes_batches = blobshape[0]*blobshape[1];
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_data_diff = bottom[0]->mutable_cpu_diff();
  Dtype* weights_diff = bottom[1]->mutable_cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weights = bottom[1]->cpu_data();
  caffe_copy(count, weights, bottom_data_diff);
  caffe_copy(count,bottom_data,weights_diff);
  top_data = top_data + count;
  bottom_data = bottom_data + count;
  for(int i=1;i<num_classes_batches;i++)
  {
      caffe_copy(count, weights, bottom_data_diff);
      caffe_add(count,weights_diff,bottom_data, weights_diff);
      bottom_data_diff = bottom_data_diff + count;
      bottom_data = bottom_data + count;
  }
}

#ifdef CPU_ONLY
STUB_GPU(DotProdLayer);
#endif

INSTANTIATE_CLASS(DotProdLayer);
REGISTER_LAYER_CLASS(DotProd);

}  // namespace caffe
