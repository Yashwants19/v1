#include "range_search.h"
#include </home/asus/mlpack/src/mlpack/methods/range_search/range_search_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetRSModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<RSModel>(identifier,
      static_cast<RSModel*>(value));
}

extern "C" void *mlpackGetRSModelPtr(const char* identifier)
{
  RSModel *modelptr = GetParamPtr<RSModel>(identifier);
  return modelptr;
}

static void RangeSearchMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackRangeSearch()
{
  RangeSearchMlpackMain();
}

