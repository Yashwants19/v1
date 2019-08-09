#include "decision_tree.h"
#include </home/asus/mlpack/src/mlpack/methods/decision_tree/decision_tree_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetDecisionTreeModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<DecisionTreeModel>(identifier,
      static_cast<DecisionTreeModel*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetDecisionTreeModelPtr(const char* identifier)
{
  DecisionTreeModel *modelptr = GetParamPtr<DecisionTreeModel>(identifier);
  return modelptr;
}

static void decision_treemlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackdecision_tree()
{
  decision_treemlpackMain();
}

