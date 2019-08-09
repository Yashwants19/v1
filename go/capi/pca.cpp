#include "pca.h"
#include </home/asus/mlpack/src/mlpack/methods/pca/pca_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void pcamlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackpca()
{
  pcamlpackMain();
}

