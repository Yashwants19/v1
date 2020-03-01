#include "adaboost.h"
#include </home/asus/mlpack/src/mlpack/methods/adaboost/adaboost_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetAdaBoostModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<AdaBoostModel>(identifier,
      static_cast<AdaBoostModel*>(value));
}

extern "C" void *mlpackGetAdaBoostModelPtr(const char* identifier)
{
  AdaBoostModel *modelptr = GetParamPtr<AdaBoostModel>(identifier);
  return modelptr;
}

static void AdaboostMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackAdaboost()
{
  AdaboostMlpackMain();
}

