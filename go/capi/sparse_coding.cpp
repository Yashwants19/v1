#include "sparse_coding.h"
#include </home/asus/mlpack/src/mlpack/methods/sparse_coding/sparse_coding_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetSparseCodingPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<SparseCoding>(identifier,
      static_cast<SparseCoding*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetSparseCodingPtr(const char* identifier)
{
  SparseCoding *modelptr = GetParamPtr<SparseCoding>(identifier);
  return modelptr;
}

static void sparse_codingmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpacksparse_coding()
{
  sparse_codingmlpackMain();
}

