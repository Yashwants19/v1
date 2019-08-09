#include "kfn.h"
#include </home/asus/mlpack/src/mlpack/methods/neighbor_search/kfn_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetKFNModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<KFNModel>(identifier,
      static_cast<KFNModel*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetKFNModelPtr(const char* identifier)
{
  KFNModel *modelptr = GetParamPtr<KFNModel>(identifier);
  return modelptr;
}

static void kfnmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackkfn()
{
  kfnmlpackMain();
}
