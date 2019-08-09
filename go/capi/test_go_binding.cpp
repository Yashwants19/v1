#include "test_go_binding.h"
#include </home/asus/mlpack/src/mlpack/bindings/go/tests/test_go_binding_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetGaussianKernelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<GaussianKernel>(identifier,
      static_cast<GaussianKernel*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetGaussianKernelPtr(const char* identifier)
{
  GaussianKernel *modelptr = GetParamPtr<GaussianKernel>(identifier);
  return modelptr;
}

static void test_go_bindingmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpacktest_go_binding()
{
  test_go_bindingmlpackMain();
}

