#include "gmm_train.h"
#include </home/asus/mlpack/src/mlpack/methods/gmm/gmm_train_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetGMMPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<GMM>(identifier,
      static_cast<GMM*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetGMMPtr(const char* identifier)
{
  GMM *modelptr = GetParamPtr<GMM>(identifier);
  return modelptr;
}

static void gmm_trainmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackgmm_train()
{
  gmm_trainmlpackMain();
}

