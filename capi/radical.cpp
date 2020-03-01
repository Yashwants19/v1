#include "radical.h"
#include </home/asus/mlpack/src/mlpack/methods/radical/radical_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void RadicalMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackRadical()
{
  RadicalMlpackMain();
}

