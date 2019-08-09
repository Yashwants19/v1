#include "hmm_viterbi.h"
#include </home/asus/mlpack/src/mlpack/methods/hmm/hmm_viterbi_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetHMMModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<HMMModel>(identifier,
      static_cast<HMMModel*>(value),
      CLI::HasParam("copy_all_inputs"));
}

extern "C" void *mlpackGetHMMModelPtr(const char* identifier)
{
  HMMModel *modelptr = GetParamPtr<HMMModel>(identifier);
  return modelptr;
}

static void hmm_viterbimlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackhmm_viterbi()
{
  hmm_viterbimlpackMain();
}

