#pragma once

#include <vector>
#include <string>
#include "nlohmann/json.hpp"
#include "ILayer.h"
#include "Matrix.h"
#include "Tensor.h"

namespace cobalt_715::nn{

//グローバルプーリング
//全結合に使えるようにする
struct GlobalAvePool2DLayer : public ILayer{
  Tensor grad_ = Tensor({1});
  Matrix a_ = Matrix(1,1);

  Tensor forward(const Tensor& input){
    #ifndef NDEBUG
    if(input.get_shape().size() != 3){
      throw std::invalid_argument(
        "GlobalAvePool2DLayer::forward: input must be 3D (C,H,W)"
      );
    }
    #endif

    grad_ = Tensor(input.get_shape());
    a_ = Matrix(input.get_shape().at(0),1);

    for(size_t in_c = 0;in_c < input.get_shape().at(0);in_c++){
      for(size_t h = 0;h < input.get_shape().at(1);h++){
        for(size_t w = 0;w < input.get_shape().at(2);w++){
          a_(in_c,0) += input.at({in_c,h,w});
        }
      }
    }

    a_ = a_ * (1.0 / (input.get_shape().at(1) * input.get_shape().at(2)));

    return Tensor::Matrix_to_Tensor(a_);
  }

  Tensor backward(const Tensor& grad_output_tensor){
    #ifndef NDEBUG
    if(grad_output_tensor.get_shape().size() != 2){
      throw std::invalid_argument(
        "GlobalAvePool2DLayer::backward: grad_output must be 2D"
      );
    }

    if(Tensor::Matrix_to_Tensor(a_).get_shape() != grad_output_tensor.get_shape()){
      throw std::invalid_argument(
        "GlobalAvePool2DLayer::backward: output gradient shape mismatch"
      );
    }
    #endif

    const size_t hw = grad_.get_shape().at(1) * grad_.get_shape().at(2);

    for(size_t in_c = 0;in_c < grad_.get_shape().at(0);in_c++){
      std::fill(grad_.data().begin() + in_c * hw,grad_.data().begin() + (in_c + 1) * hw,grad_output_tensor.at({in_c,0}) / hw);
    }

    return grad_;
  }

  void step(double lr,int batch_size=64){}

  std::string get_type() const{
    return "GlobalAvePoll2DLayer";
  }

  nlohmann::ordered_json to_json() const{
    return nlohmann::ordered_json();
  }

  void load_json(nlohmann::ordered_json j){}

  void random_init(std::mt19937 &gen){}
};

}//namespace cobalt_715::nn