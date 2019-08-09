#include "approx_kfn.h"
#include </home/asus/mlpack/src/mlpack/methods/approx_kfn/approx_kfn_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetApproxKFNModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<ApproxKFNModel>(identifier,
      static_cast<ApproxKFNModel*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetApproxKFNModelPtr(const char* identifier)
{
  ApproxKFNModel *modelptr = GetParamPtr<ApproxKFNModel>(identifier);
  return modelptr;
}

static void approx_kfnmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackapprox_kfn()
{
  approx_kfnmlpackMain();
}
