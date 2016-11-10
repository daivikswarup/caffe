#include <cfloat>
#include <vector>

#include "caffe/layers/dotprod_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void DotProdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const vector<int> blobshape = bottom[0]->shape();
    const int count = blobshape[2]*blobshape[3];//Number of pixels
    const int num_classes_batches = blobshape[0]*blobshape[1];
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* weights = bottom[1]->gpu_data();
    for(int i=0;i<num_classes_batches;i++)
    {
        caffe_gpu_mul(count, bottom_data, weights, top_data);
        top_data = top_data + count;
        bottom_data = bottom_data + count;
    }
  }
}


template <typename Dtype>
void DotProdLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


    const vector<int> blobshape = bottom[0]->shape();
    const int count = blobshape[2]*blobshape[3];//Number of pixels
    const int num_classes_batches = blobshape[0]*blobshape[1];
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* bottom_data_diff = bottom[0]->mutable_gpu_diff();
    Dtype* weights_diff = bottom[1]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* weights = bottom[1]->gpu_data();
    caffe_copy(count, weights, bottom_data_diff);
    caffe_copy(count,bottom_data,weights_diff);
    top_data = top_data + count;
    bottom_data = bottom_data + count;
    for(int i=1;i<num_classes_batches;i++)
    {
        caffe_copy(count, weights, bottom_data_diff);
        caffe_gpu_add(count,weights_diff,bottom_data, weights_diff);
        bottom_data_diff = bottom_data_diff + count;
        bottom_data = bottom_data + count;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DotProdLayer);

}  // namespace caffe
