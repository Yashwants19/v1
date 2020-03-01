#include "preprocess_describe.h"
#include </home/asus/mlpack/src/mlpack/methods/preprocess/preprocess_describe_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void PreprocessDescribeMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackPreprocessDescribe()
{
  PreprocessDescribeMlpackMain();
}

