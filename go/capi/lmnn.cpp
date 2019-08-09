#include "lmnn.h"
#include </home/asus/mlpack/src/mlpack/methods/lmnn/lmnn_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void lmnnmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpacklmnn()
{
  lmnnmlpackMain();
}

