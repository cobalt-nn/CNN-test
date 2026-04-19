#include <iostream>
#include <vector>
#include "nn/Matrix.h"
#include "nn/Trainer.h"
#include "nn/Activation.h"
#include "nn/Acts.h"
#include "nn/Tensor.h"
#include "nn/Conv2DLayer.h"

using namespace cobalt_715::nn;

int main(){
  std::vector<double> v(100);
  for(size_t i = 0;i < v.size();i++){
    v[i] = i;
  }

  Tensor t({2,5,10},v);

  Conv2DLayer layer(2,3,3,3);

  std::cout << t.to_string() << "\n" << layer.to_string() << std::endl;

  std::cout << layer.forward(t).to_string() << std::endl;

  return 0;
}
