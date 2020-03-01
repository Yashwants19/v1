#include "kernel_pca.h"
#include </home/asus/mlpack/src/mlpack/methods/kernel_pca/kernel_pca_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void KernelPcaMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackKernelPca()
{
  KernelPcaMlpackMain();
}

