#include "nmf.h"
#include </home/asus/mlpack/src/mlpack/methods/nmf/nmf_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void NmfMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackNmf()
{
  NmfMlpackMain();
}

