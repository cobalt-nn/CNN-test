#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "Matrix.h"

namespace cobalt_715::nn{

//任意次元テンソル
class Tensor{
private:
  std::vector<size_t> shape_;//各次元の要素数
  std::vector<size_t> stride_;//各次元にジャンプするまでに必要な数
  std::vector<double> data_;//data

  //shape_,stride_,data_の関係が合うかどうかとオーバーフローの確認をする
  void check_invariants() const{
    #ifndef NDEBUG
    //shape_とstride_のサイズが同じかどうか確認する
    if(shape_.size() != stride_.size()) throw std::logic_error("Tensor invariant violation: shape and stride size mismatch");

    //data_のサイズがshape_と合っているかとオーバーフローしていないか確認する
    size_t data_size = 1;
    for(size_t i = 0;i < shape_.size();i++){
      size_t dim = shape_[i];

      //0になっていないか確認する
      if(dim == 0) throw std::invalid_argument("Tensor shape contains zero dimension");

      //オーバーフローを確認する
      if(dim > SIZE_MAX / data_size) throw std::overflow_error("Tensor size overflow");

      data_size *= dim;
    }
    if(data_.size() != data_size) throw std::logic_error("Tensor invariant violation: data size mismatch");

    //stride_がshape_と合っているか確認する
    if(stride_ != make_stride()) throw std::logic_error("Tensor invariant violation: invalid stride");
    #endif
  }

  //indexが正しいか確認する
  void check_index(const std::vector<size_t>& a) const{
    #ifndef NDEBUG
    if(a.size() != shape_.size()){
      throw std::invalid_argument(
        "Tensor index dimension mismatch: expected " +
        std::to_string(shape_.size()) +
        ", got " +
        std::to_string(a.size())
      );
    }

    for(size_t i = 0;i < shape_.size();i++){
      if(a[i] >= shape_[i]){
        throw std::out_of_range(
          "Tensor index out of bounds at dim " +
          std::to_string(i) +
          ": index=" + std::to_string(a[i]) +
          ", size=" + std::to_string(shape_[i])
        );
      }
    }
    #endif
  }

  //shape_を元にstride_を作る
  std::vector<size_t> make_stride() const{
    std::vector<size_t> str(shape_.size());
    size_t stride_size = 1;
    for(size_t i = shape_.size();i-- > 0;){
      str[i] = stride_size;
      stride_size *= shape_[i];
    }
    return str;
  }

public:
  double& at(const std::vector<size_t>& a){
    check_index(a);
    //check_invariants();
    size_t index = 0;
    for(size_t i = 0;i < a.size();i++){
      index += a[i] * stride_[i];
    }
    return data_[index];
  }

  const double& at(const std::vector<size_t>& a) const{
    check_index(a);
    //check_invariants();
    size_t index = 0;
    for(size_t i = 0;i < a.size();i++){
      index += a[i] * stride_[i];
    }
    return data_[index];
  }

  const std::vector<size_t>& get_shape() const{
    return shape_;
  }

  //参照を返しているため気を付けること
  std::vector<double>& data(){return data_;};
  const std::vector<double>& data() const{return data_;}

  //スカラー倍　a * dの結果をoutに入れる
  static void scale(const Tensor &a,double d,Tensor &out){
    #ifndef NDEBUG
    if(a.get_shape() != out.get_shape()) throw std::invalid_argument("Tensor::scale dimension mismatch");
    #endif

    const double *ad = a.data().data();
    double *od = out.data().data();

    for(size_t i = 0;i < a.data().size();i++){
      od[i] = ad[i] * d;
    }
  }

  Tensor operator*(double d) const{
    Tensor out(shape_);
    scale(*this,d,out);
    return out;
  }

  Tensor& operator*=(double d){
    scale(*this,d,*this);
    return *this;
  }

  //要素積
  static void hadamard(const Tensor &a,const Tensor &b,Tensor &out){
    #ifndef NDEBUG
    if(a.get_shape() != out.get_shape() || a.get_shape() != b.get_shape()) throw std::invalid_argument("Tensor::hadamard dimension mismatch");
    #endif

    const double *ad = a.data().data();
    const double *bd = b.data().data();
    double *od = out.data().data();

    for(size_t i = 0;i < a.data().size();i++){
      od[i] = ad[i] * bd[i];
    }
  }

  Tensor hadamard(const Tensor &rhs) const{
    Tensor out(shape_);
    hadamard(*this,rhs,out);
    return out;
  }

  //a + bの結果をoutに入れる
  static void add(const Tensor &a,const Tensor &b,Tensor &out){
    #ifndef NDEBUG
    if(a.get_shape() != out.get_shape() || a.get_shape() != b.get_shape()) throw std::invalid_argument("Tensor::add dimension mismatch");
    #endif

    const double *ad = a.data().data();
    const double *bd = b.data().data();
    double *od = out.data().data();

    for(size_t i = 0;i < a.data().size();i++){
      od[i] = ad[i] + bd[i];
    }
  }

  Tensor operator+(const Tensor &rhs) const{
    Tensor out(shape_);
    add(*this,rhs,out);
    return out;
  }

  Tensor& operator+=(const Tensor &rhs){
    add(*this,rhs,*this);
    return *this;
  }

  //a - bの結果をoutに入れる
  static void subtract(const Tensor &a,const Tensor &b,Tensor &out){
    #ifndef NDEBUG
    if(a.get_shape() != out.get_shape() || a.get_shape() != b.get_shape()) throw std::invalid_argument("Tensor::subtract dimension mismatch");
    #endif

    const double *ad = a.data().data();
    const double *bd = b.data().data();
    double *od = out.data().data();

    for(size_t i = 0;i < a.data().size();i++){
      od[i] = ad[i] - bd[i];
    }
  }

  Tensor operator-(const Tensor &rhs) const{
    Tensor out(shape_);
    subtract(*this,rhs,out);
    return out;
  }

  Tensor& operator-=(const Tensor &rhs){
    subtract(*this,rhs,*this);
    return *this;
  }

  std::string to_string(const int indent_size = 2) const{
    std::string s = "//shape = ";

    for(size_t u:shape_) s += std::to_string(u) + " ";

    s += "\n";

    std::vector<size_t> index(shape_.size(),0);

    size_t dim = 0;

    to_string_recursive(s,index,dim,indent_size);

    return s;
  }

  void to_string_recursive(std::string &s,std::vector<size_t> &index,size_t dim,const int indent_size = 2) const{
    s += std::string(indent_size * dim,' ') + "{";

    for(size_t i = 0;i < shape_.at(dim);i++){
      index.at(dim) = i;

      if(dim == shape_.size() - 1){
        s += " " + std::to_string(at(index)) + " ";
      }else{
        s += "\n";
        to_string_recursive(s,index,dim + 1,indent_size);
      }

      if(i != shape_.at(dim) - 1) s += ",";
    }

    if(dim == shape_.size() - 1){
      s += "}";
    }else{
      s += "\n" + std::string(indent_size * dim,' ') + "}";
    }
  }

  std::vector<Matrix> to_Matrix() const{
    std::vector<Matrix> matrixs;

    if(shape_.size() == 1){
      matrixs.push_back(Matrix(1,shape_.at(0),data_));
      return matrixs;
    }

    size_t row = shape_[shape_.size() - 2];
    size_t col = shape_[shape_.size() - 1];

    size_t matrix_data_size = row * col;

    for(size_t i = 0;i < data_.size();i += matrix_data_size){
      std::vector<double> d(data_.data() + i,data_.data() + i + matrix_data_size);
      Matrix m(row,col,d);
      matrixs.push_back(m);
    }
    return matrixs;
  }

  static Tensor Matrix_to_Tensor(const Matrix &m){
    return Tensor({(size_t)m.rows(),(size_t)m.cols()},m.data());
  }

  //コンストラクタ
  Tensor(const std::vector<size_t> &shape) : shape_(shape),stride_(make_stride()){
    size_t data_size = 1;
    for(size_t u:shape_){
      data_size *= u;
    }
    data_.resize(data_size);
    check_invariants();
  }

  Tensor(const std::vector<size_t> &shape,const std::vector<double> &data) : shape_(shape),stride_(make_stride()),data_(data){
    check_invariants();
  }
};

}//namespace cobalt_715::nn