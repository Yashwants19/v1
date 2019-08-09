#include "decision_stump.h"
#include </home/asus/mlpack/src/mlpack/methods/decision_stump/decision_stump_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetDSModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<DSModel>(identifier,
      static_cast<DSModel*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetDSModelPtr(const char* identifier)
{
  DSModel *modelptr = GetParamPtr<DSModel>(identifier);
  return modelptr;
}

static void decision_stumpmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackdecision_stump()
{
  decision_stumpmlpackMain();
}

