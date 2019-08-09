#include "knn.h"
#include </home/asus/mlpack/src/mlpack/methods/neighbor_search/knn_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetKNNModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<KNNModel>(identifier,
      static_cast<KNNModel*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetKNNModelPtr(const char* identifier)
{
  KNNModel *modelptr = GetParamPtr<KNNModel>(identifier);
  return modelptr;
}

static void knnmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackknn()
{
  knnmlpackMain();
}

