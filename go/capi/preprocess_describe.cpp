#include "preprocess_describe.h"
#include </home/asus/mlpack/src/mlpack/methods/preprocess/preprocess_describe_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void preprocess_describemlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackpreprocess_describe()
{
  preprocess_describemlpackMain();
}

