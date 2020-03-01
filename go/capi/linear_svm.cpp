#include "linear_svm.h"
#include </home/asus/mlpack/src/mlpack/methods/linear_svm/linear_svm_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetLinearSVMModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<LinearSVMModel>(identifier,
      static_cast<LinearSVMModel*>(value));
}

extern "C" void *mlpackGetLinearSVMModelPtr(const char* identifier)
{
  LinearSVMModel *modelptr = GetParamPtr<LinearSVMModel>(identifier);
  return modelptr;
}

static void LinearSvmMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackLinearSvm()
{
  LinearSvmMlpackMain();
}

