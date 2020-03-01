#include "mean_shift.h"
#include </home/asus/mlpack/src/mlpack/methods/mean_shift/mean_shift_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void MeanShiftMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackMeanShift()
{
  MeanShiftMlpackMain();
}

