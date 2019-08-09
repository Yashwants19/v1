#include "det.h"
#include </home/asus/mlpack/src/mlpack/methods/det/det_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetDTreePtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<DTree<>>(identifier,
      static_cast<DTree<>*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetDTreePtr(const char* identifier)
{
  DTree<> *modelptr = GetParamPtr<DTree<>>(identifier);
  return modelptr;
}

static void detmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackdet()
{
  detmlpackMain();
}

