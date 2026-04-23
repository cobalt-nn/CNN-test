#pragma once

//#include <iostream>
#include <stdexcept>
#include <string>
#include <random>
#include "nlohmann/json.hpp"
#include "ILayer.h"
#include "Activation.h"
#include "Acts.h"

namespace cobalt_715::nn{

//2D畳み込みニューラルネットワーク
//ミニバッチ非対応
struct Conv2DLayer : public ILayer{
  enum class PaddingType{
    Valid,//パディングなし
    Same//出力サイズ維持
  };

  const size_t in_channels_,out_channels_;//チャンネルサイズ
  const size_t kh_,kw_;//カーネルサイズ
  const PaddingType type_;
  Tensor input_;////逆伝播で必要なため順伝播時この層への入力を保持する
  Tensor W_,b_;//重み、バイアス
  Tensor z_,a_;//活性化前、活性化後
  Tensor dW_,db_;//重みの微分、バイアスの微分
  Tensor delta_,grad_;//この層での誤差、次の層に渡す勾配

  const Activation *act_ = &activations::LeakyReLU;//活性化関数とその微分。デフォルトではLeakyReLU

  //順伝播
  //前層の出力を受け取る
  Tensor forward(const Tensor& input) override{
    #ifndef NDEBUG
    if(input.get_shape().size() != 3){
      throw std::invalid_argument(
        "Conv2DLayer::forward: input must be 3D (C,H,W)"
      );
    }

    if(input.get_shape().at(0) != in_channels_){
      throw std::invalid_argument(
        "Conv2DLayer::forward: channel count mismatch"
      );
    }
    #endif

    input_ = input;

    //出力サイズ
    size_t zh = 0;
    size_t zw = 0;

    switch(type_){
      case PaddingType::Same:
        zh = input_.get_shape().at(1);
        zw = input_.get_shape().at(2);
        break;
      case PaddingType::Valid:
        zh = input_.get_shape().at(1) - kh_ + 1;
        zw = input_.get_shape().at(2) - kw_ + 1;
        break;
    }

    z_ = Tensor({out_channels_,zh,zw});
    a_ = Tensor({out_channels_,zh,zw});

    std::vector<size_t> W_index = {0,0,0,0};

    //i,jはz_,a_の中心とする
    for(size_t i = 0;i < zh;i++){//std::cout << 1 << std::endl;
      for(size_t j = 0;j < zw;j++){//std::cout << 2 << std::endl;
        for(size_t out_c = 0;out_c < out_channels_;out_c++){//std::cout << 3 << std::endl;
          W_index[0] = out_c;
          double total = b_.at({out_c});

          for(size_t in_c = 0;in_c < in_channels_;in_c++){//std::cout << 4 << std::endl;
            W_index[1] = in_c;
            for(size_t kh = 0;kh < kh_;kh++){//std::cout << 5 << std::endl;
              W_index[2] = kh;
              for(size_t kw = 0;kw < kw_;kw++){//std::cout << 6 << std::endl;
                W_index[3] = kw;

                size_t in_row = i + kh;
                size_t in_col = j + kw;

                if(type_ == PaddingType::Same){
                  in_row -= kh_ / 2;
                  in_col -= kw_ / 2;
                }

                if(in_row >= input.get_shape().at(1) || in_col >= input.get_shape().at(2) || i + kh < in_row || j + kw < in_col) continue;

                total += input.at({in_c,in_row,in_col}) * W_.at(W_index);
              }
            }
          }

          z_.at({out_c,i,j}) = total;
          a_.at({out_c,i,j}) = act_->act(total);
        }
      }
    }

    return a_;
  }

  Tensor backward(const Tensor& grad_output_tensor){
    #ifndef NDEBUG
    if(grad_output_tensor.get_shape().size() != 3){
      throw std::invalid_argument(
        "Conv2DLayer::backward: grad_output must be 3D"
      );
    }

    if(a_.get_shape() != grad_output_tensor.get_shape()){
      throw std::invalid_argument(
        "Conv2DLayer::backward: output gradient shape mismatch"
      );
    }
    #endif

    grad_ = Tensor(input_.get_shape());

    std::fill(db_.data().begin(),db_.data().end(),0.0);
    std::fill(dW_.data().begin(),dW_.data().end(),0.0);
    
    for(size_t i = 0;i < a_.get_shape().at(1);i++){
      for(size_t j = 0;j < a_.get_shape().at(2);j++){

        for(size_t out_c = 0;out_c < out_channels_;out_c++){
          double delta = grad_output_tensor.at({out_c,i,j}) * act_->d_act(z_.at({out_c,i,j}),a_.at({out_c,i,j}));
          db_.at({out_c}) += delta;

          for(size_t in_c = 0;in_c < in_channels_;in_c++){

            for(size_t kh = 0;kh < kh_;kh++){
              for(size_t kw = 0;kw < kw_;kw++){

                size_t in_row = i + kh;
                size_t in_col = j + kw;

                if(type_ == PaddingType::Same){
                  in_row -= kh_ / 2;
                  in_col -= kw_ / 2;
                }

                if(in_row >= input_.get_shape().at(1) || in_col >= input_.get_shape().at(2) || i + kh < in_row || j + kw < in_col) continue;

                dW_.at({out_c,in_c,kh,kw}) += delta * input_.at({in_c,in_row,in_col});
                grad_.at({in_c,in_row,in_col}) += delta * W_.at({out_c,in_c,kh,kw});
              }
            }
          }
        }
      }
    }

    return grad_;
  }

  void step(double lr,int batch_size=64){
    W_ = W_ - dW_ * lr;
    b_ = b_ - db_ * lr;
  }

  std::string get_type() const{
    return "Conv2DLayer";
  }

  std::string to_string() const override{
    std::string s;
    s += "activation " + act_->name;
    s += "\nW\n";
    s += W_.to_string() + "\nb\n";
    s += b_.to_string();
    return s;
  }

  virtual nlohmann::ordered_json to_json() const{
    return nlohmann::ordered_json();
  }

  virtual void load_json(nlohmann::ordered_json j){
  }

  virtual void random_init(std::mt19937 &gen){
    double limit = sqrt(2.0 / (in_channels_ * kh_ * kw_));
    std::uniform_real_distribution<double> dist(-limit,limit);
    for(double &d:W_.data()){
      d = dist(gen);
    }
  }

  Conv2DLayer(size_t in_channels,size_t out_channels,size_t kh,size_t kw)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kh_(kh),
      kw_(kw),
      type_(PaddingType::Same),
      input_({1}),
      W_({out_channels,in_channels,kh,kw}),
      b_({out_channels}),
      z_({1}),
      a_({1}),
      dW_(W_.get_shape()),
      db_(b_.get_shape()),
      delta_({1}),
      grad_({1}){

    std::mt19937 gen(0);
    random_init(gen);
  }
};

}//namespace cobalt_715::nn