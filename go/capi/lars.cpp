#include "lars.h"
#include </home/asus/mlpack/src/mlpack/methods/lars/lars_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetLARSPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<LARS>(identifier,
      static_cast<LARS*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetLARSPtr(const char* identifier)
{
  LARS *modelptr = GetParamPtr<LARS>(identifier);
  return modelptr;
}

static void larsmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpacklars()
{
  larsmlpackMain();
}

