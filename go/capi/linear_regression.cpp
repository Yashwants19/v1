#include "linear_regression.h"
#include </home/asus/mlpack/src/mlpack/methods/linear_regression/linear_regression_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetLinearRegressionPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<LinearRegression>(identifier,
      static_cast<LinearRegression*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetLinearRegressionPtr(const char* identifier)
{
  LinearRegression *modelptr = GetParamPtr<LinearRegression>(identifier);
  return modelptr;
}

static void linear_regressionmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpacklinear_regression()
{
  linear_regressionmlpackMain();
}

