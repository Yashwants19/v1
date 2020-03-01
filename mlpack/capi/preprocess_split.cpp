#include "preprocess_split.h"
#include </home/asus/mlpack/src/mlpack/methods/preprocess/preprocess_split_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void PreprocessSplitMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackPreprocessSplit()
{
  PreprocessSplitMlpackMain();
}

