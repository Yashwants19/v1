#include "nbc.h"
#include </home/asus/mlpack/src/mlpack/methods/naive_bayes/nbc_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetNBCModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<NBCModel>(identifier,
      static_cast<NBCModel*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetNBCModelPtr(const char* identifier)
{
  NBCModel *modelptr = GetParamPtr<NBCModel>(identifier);
  return modelptr;
}

static void nbcmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpacknbc()
{
  nbcmlpackMain();
}

