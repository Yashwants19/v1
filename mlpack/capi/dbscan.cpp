#include "dbscan.h"
#include </home/asus/mlpack/src/mlpack/methods/dbscan/dbscan_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void DbscanMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackDbscan()
{
  DbscanMlpackMain();
}

