#include "local_coordinate_coding.h"
#include </home/asus/mlpack/src/mlpack/methods/local_coordinate_coding/local_coordinate_coding_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetLocalCoordinateCodingPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<LocalCoordinateCoding>(identifier,
      static_cast<LocalCoordinateCoding*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetLocalCoordinateCodingPtr(const char* identifier)
{
  LocalCoordinateCoding *modelptr = GetParamPtr<LocalCoordinateCoding>(identifier);
  return modelptr;
}

static void local_coordinate_codingmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpacklocal_coordinate_coding()
{
  local_coordinate_codingmlpackMain();
}

