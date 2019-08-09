#include "softmax_regression.h"
#include </home/asus/mlpack/src/mlpack/methods/softmax_regression/softmax_regression_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetSoftmaxRegressionPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<SoftmaxRegression>(identifier,
      static_cast<SoftmaxRegression*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetSoftmaxRegressionPtr(const char* identifier)
{
  SoftmaxRegression *modelptr = GetParamPtr<SoftmaxRegression>(identifier);
  return modelptr;
}

static void softmax_regressionmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpacksoftmax_regression()
{
  softmax_regressionmlpackMain();
}

