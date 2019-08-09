#include "perceptron.h"
#include </home/asus/mlpack/src/mlpack/methods/perceptron/perceptron_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetPerceptronModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<PerceptronModel>(identifier,
      static_cast<PerceptronModel*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetPerceptronModelPtr(const char* identifier)
{
  PerceptronModel *modelptr = GetParamPtr<PerceptronModel>(identifier);
  return modelptr;
}

static void perceptronmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackperceptron()
{
  perceptronmlpackMain();
}

