#include "random_forest.h"
#include </home/asus/mlpack/src/mlpack/methods/random_forest/random_forest_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetRandomForestModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<RandomForestModel>(identifier,
      static_cast<RandomForestModel*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetRandomForestModelPtr(const char* identifier)
{
  RandomForestModel *modelptr = GetParamPtr<RandomForestModel>(identifier);
  return modelptr;
}

static void random_forestmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackrandom_forest()
{
  random_forestmlpackMain();
}

