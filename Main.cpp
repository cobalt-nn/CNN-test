#include <iostream>
#include <vector>
#include "nn/Matrix.h"
#include "nn/Trainer.h"
#include "nn/Activation.h"
#include "nn/Acts.h"
#include "nn/Tensor.h"
#include "nn/Conv2DLayer.h"

#include "nn/GlobalAvePool2DLayer.h"
#include "nn/Pool2DLayer.h"

using namespace cobalt_715::nn;

int main(){
  std::vector<double> v(100);
  for(size_t i = 0;i < v.size();i++){
    v[i] = i;
  }

  Tensor t({2,5,10},v);

  Conv2DLayer layer(2,3,3,3);
  Pool2DLayer pool(2,2);
  GlobalAvePoll2DLayer glo;

  std::cout << t.to_string() << "\n" << layer.to_string() << std::endl;

  Tensor z1 = layer.forward(t);

  std::cout << z1.to_string() << std::endl;

  Tensor z2 = pool.forward(z1);

  std::cout << z2.to_string() << std::endl;

  Tensor z3 = glo.forward(z2);

  std::cout << z3.to_string() << std::endl;

  return 0;
}
