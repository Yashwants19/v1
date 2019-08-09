#include "emst.h"
#include </home/asus/mlpack/src/mlpack/methods/emst/emst_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void emstmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackemst()
{
  emstmlpackMain();
}

