#include "nca.h"
#include </home/asus/mlpack/src/mlpack/methods/nca/nca_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void ncamlpackMain()
{
  mlpackMain();
}

extern "C" void mlpacknca()
{
  ncamlpackMain();
}

