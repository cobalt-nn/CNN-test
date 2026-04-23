#pragma once

#include <vector>
#include <string>
#include "nlohmann/json.hpp"
#include "ILayer.h"
#include "Tensor.h"

namespace cobalt_715::nn{

//プーリング
struct Pool2DLayer : public ILayer{
  enum class PoolType{
    Max,
    Ave
  };

  const size_t h_,w_;
  const PoolType type_;
  Tensor a_ = Tensor({1});
  Tensor grad_ = Tensor({1});

  Tensor forward(const Tensor& input){
    #ifndef NDEBUG
    if(input.get_shape().size() != 3){
      throw std::invalid_argument(
        "Pool2DLayer::forward: input must be 3D (C,H,W)"
      );
    }
    #endif

    const std::vector<size_t> input_shape = input.get_shape();

    a_ = Tensor({input_shape.at(0),(input_shape.at(1) - h_) / h_ + 1,(input_shape.at(2) - w_) / w_ + 1});
    grad_ = Tensor(input.get_shape());

    std::vector<size_t> grad_max_index(3);

    switch(type_){
      case PoolType::Max:
        std::fill(grad_.data().begin(),grad_.data().end(),0);
        break;
      case PoolType::Ave:
        std::fill(grad_.data().begin(),grad_.data().end(),1.0 / (h_ * w_));
        break;
    }

    for(size_t in_c = 0;in_c < input_shape.at(0);in_c++){
      grad_max_index.at(0) = in_c;
      for(size_t a_row = 0;a_row < a_.get_shape().at(1);a_row++){
        for(size_t a_col = 0;a_col < a_.get_shape().at(2);a_col++){
          double num = -std::numeric_limits<double>::infinity();
          for(size_t h = 0;h < h_;h++){
            for(size_t w = 0;w < w_;w++){
              if(a_row * h_ + h >= input_shape.at(1) || a_col * w_ + w >= input_shape.at(2)) continue;
              switch(type_){
                case PoolType::Max:
                  if(input.at({in_c,a_row * h_ + h,a_col * w_ + w}) > num){
                    num = input.at({in_c,a_row * h_ + h,a_col * w_ + w});
                    grad_max_index.at(1) = a_row * h_ + h;
                    grad_max_index.at(2) = a_col * w_ + w;
                  }
                  break;
                case PoolType::Ave:
                  num += input.at({in_c,a_row * h_ + h,a_col * w_ + w});
                  break;
              }
            }
          }
          a_.at({in_c,a_row,a_col}) = num;
          if(type_ == PoolType::Max) grad_.at(grad_max_index) = 1;
        }
      }
    }

    return a_;
  }

  Tensor backward(const Tensor& grad_output_tensor){
    if(a_.get_shape() != grad_output_tensor.get_shape()){
      throw std::invalid_argument(
        "Conv2DLayer::backward: output gradient shape mismatch"
      );
    }
    for(size_t in_c = 0;in_c < a_.get_shape().at(0);in_c++){
      for(size_t a_row = 0;a_row < a_.get_shape().at(1);a_row++){
        for(size_t a_col = 0;a_col < a_.get_shape().at(2);a_col++){
          for(size_t h = 0;h < h_;h++){
            for(size_t w = 0;w < w_;w++){
              if(a_row * h_ + h >= grad_.get_shape().at(1) || a_col * w_ + w >= grad_.get_shape().at(2)) continue;
              grad_.at({in_c,a_row * h_ + h,a_col * w_ + w}) *= grad_output_tensor.at({in_c,a_row,a_col});
            }
          }
        }
      }
    }

    return grad_;
  }

  void step(double lr,int batch_size=64){}

  std::string get_type() const{
    return "Pool2DLayer";
  }

  nlohmann::ordered_json to_json() const{
    return nlohmann::ordered_json();
  }

  void load_json(nlohmann::ordered_json j){}

  void random_init(std::mt19937 &gen){}

  Pool2DLayer(size_t h,size_t w) : h_(h),w_(w),type_(PoolType::Max){}
};

}//namespace cobalt_715::nn