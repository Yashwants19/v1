#include "logistic_regression.h"
#include </home/asus/mlpack/src/mlpack/methods/logistic_regression/logistic_regression_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetLogisticRegressionPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<LogisticRegression<>>(identifier,
      static_cast<LogisticRegression<>*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetLogisticRegressionPtr(const char* identifier)
{
  LogisticRegression<> *modelptr = GetParamPtr<LogisticRegression<>>(identifier);
  return modelptr;
}

static void logistic_regressionmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpacklogistic_regression()
{
  logistic_regressionmlpackMain();
}

