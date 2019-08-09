#include "kmeans.h"
#include </home/asus/mlpack/src/mlpack/methods/kmeans/kmeans_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void kmeansmlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackkmeans()
{
  kmeansmlpackMain();
}

