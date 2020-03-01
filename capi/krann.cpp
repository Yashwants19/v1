#include "krann.h"
#include </home/asus/mlpack/src/mlpack/methods/rann/krann_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetRANNModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<RANNModel>(identifier,
      static_cast<RANNModel*>(value));
}

extern "C" void *mlpackGetRANNModelPtr(const char* identifier)
{
  RANNModel *modelptr = GetParamPtr<RANNModel>(identifier);
  return modelptr;
}

static void KrannMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackKrann()
{
  KrannMlpackMain();
}
