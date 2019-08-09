package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_hmm_viterbi
#include <capi/hmm_viterbi.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type Hmm_viterbiOptionalParam struct {
    Copy_all_inputs bool
    Verbose bool
}

func InitializeHmm_viterbi() *Hmm_viterbiOptionalParam {
  return &Hmm_viterbiOptionalParam{
    Copy_all_inputs: false,
    Verbose: false,
  }
}

type HMMModel struct {
 mem unsafe.Pointer
}

func (m *HMMModel) allocHMMModel(identifier string) {
 m.mem = C.mlpackGetHMMModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *HMMModel) getHMMModel(identifier string) {
 m.allocHMMModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setHMMModel(identifier string, ptr *HMMModel) {
 C.mlpackSetHMMModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This utility takes an already-trained HMM, specified as 'input_model', and
  evaluates the most probable hidden state sequence of a given sequence of
  observations (specified as ''input', using the Viterbi algorithm.  The
  computed state sequence may be saved using the 'output' output parameter.
  
  For example, to predict the state sequence of the observations obs using the
  HMM hmm, storing the predicted state sequence to states, the following command
  could be used:
  
  param := InitializeHmm_viterbi()
  states := Hmm_viterbi(obs, hmm, )


  Input parameters:

   - input (mat.Dense): Matrix containing observations,
   - input_model (HMMModel): Trained HMM to use.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): File to save predicted state sequence to.

*/
func Hmm_viterbi(input *mat.Dense, input_model *HMMModel, param *Hmm_viterbiOptionalParam) (*mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Hidden Markov Model (HMM) Viterbi State Prediction")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  GonumToArmaMat("input", input)
  SetPassed("input")

  // Detect if the parameter was passed; set if so.
  setHMMModel("input_model", input_model)
  SetPassed("input_model")

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("output")

  // Call the mlpack program.
  C.mlpackhmm_viterbi()

  // Initialize result variable and get output.
  var output_ptr mlpackArma
  output := output_ptr.ArmaToGonumUmat("output")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output
}
