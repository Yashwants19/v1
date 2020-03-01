#include "hoeffding_tree.h"
#include </home/asus/mlpack/src/mlpack/methods/hoeffding_trees/hoeffding_tree_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetHoeffdingTreeModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<HoeffdingTreeModel>(identifier,
      static_cast<HoeffdingTreeModel*>(value));
}

extern "C" void *mlpackGetHoeffdingTreeModelPtr(const char* identifier)
{
  HoeffdingTreeModel *modelptr = GetParamPtr<HoeffdingTreeModel>(identifier);
  return modelptr;
}

static void HoeffdingTreeMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackHoeffdingTree()
{
  HoeffdingTreeMlpackMain();
}

