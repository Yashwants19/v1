#include "preprocess_binarize.h"
#include </home/asus/mlpack/src/mlpack/methods/preprocess/preprocess_binarize_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void preprocess_binarizemlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackpreprocess_binarize()
{
  preprocess_binarizemlpackMain();
}

