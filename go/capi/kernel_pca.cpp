#include "kernel_pca.h"
#include </home/asus/mlpack/src/mlpack/methods/kernel_pca/kernel_pca_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void kernel_pcamlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackkernel_pca()
{
  kernel_pcamlpackMain();
}

