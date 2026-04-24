#include <iostream>
#include <vector>
#include <random>
#include "nn/Matrix.h"
#include "nn/Trainer.h"
#include "nn/Activation.h"
#include "nn/Acts.h"
#include "nn/Tensor.h"

#include "nn/ILayer.h"
#include "nn/Conv2DLayer.h"
#include "nn/GlobalAvePool2DLayer.h"
#include "nn/Pool2DLayer.h"
#include "nn/DenseLayer.h"

#include "data/MNISTLoader.h"

using namespace cobalt_715::nn;

Matrix setTenMatrix(int i){
  Matrix ten(10,1);
  ten(i,0) = 1;
  return ten;
}

int main(){
  MNISTLoader mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
  Trainer t("nn/models/model.json");

  t.layers.push_back(std::make_unique<Conv2DLayer>(1,8,3,3));
  t.layers.push_back(std::make_unique<Pool2DLayer>(2,2));
  t.layers.push_back(std::make_unique<Conv2DLayer>(8,16,3,3));
  t.layers.push_back(std::make_unique<Pool2DLayer>(2,2));
  t.layers.push_back(std::make_unique<Conv2DLayer>(16,32,3,3));
  t.layers.push_back(std::make_unique<GlobalAvePool2DLayer>());
  t.layers.push_back(std::make_unique<DenseLayer>(32,10));

  std::mt19937 gen(0);

  for(auto &l:t.layers){
    l->random_init(gen);
  }

  const int MNIST_size = 60000;

  for(size_t i = 0;i < MNIST_size;i++){
    Tensor input = Tensor({1,28,28},mnist.getImage(i));
    Tensor target = Tensor::Matrix_to_Tensor(setTenMatrix(mnist.getLabel(i)));

    Tensor z = t.forward_network(input);

    std::cout << i << " times\n" << mnist.getLabel(i) << "\n" << z.to_string() << std::endl;

    t.backward_network(z - target);

    t.step_network();
  }

  return 0;
}
