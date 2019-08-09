package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_preprocess_binarize
#include <capi/preprocess_binarize.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type Preprocess_binarizeOptionalParam struct {
    Copy_all_inputs bool
    Dimension int
    Threshold float64
    Verbose bool
}

func InitializePreprocess_binarize() *Preprocess_binarizeOptionalParam {
  return &Preprocess_binarizeOptionalParam{
    Copy_all_inputs: false,
    Dimension: 0,
    Threshold: 0,
    Verbose: false,
  }
}

/*
  This utility takes a dataset and binarizes the variables into either 0 or 1
  given threshold. User can apply binarization on a dimension or the whole
  dataset.  The dimension to apply binarization to can be specified using the
  'dimension' parameter; if left unspecified, every dimension will be binarized.
   The threshold for binarization can also be specified with the 'threshold'
  parameter; the default threshold is 0.0.
  
  The binarized matrix may be saved with the 'output' output parameter.
  
  For example, if we want to set all variables greater than 5 in the dataset X
  to 1 and variables less than or equal to 5.0 to 0, and save the result to Y,
  we could run
  
  param := InitializePreprocess_binarize()
  param.Threshold = 5
  Y := Preprocess_binarize(X, param)
  
  But if we want to apply this to only the first (0th) dimension of X,  we could
  instead run
  
  param := InitializePreprocess_binarize()
  param.Threshold = 5
  param.Dimension = 0
  Y := Preprocess_binarize(X, param)


  Input parameters:

   - input (mat.Dense): Input data matrix.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - dimension (int): Dimension to apply the binarization. If not set, the
        program will binarize every dimension by default.  Default value 0.
   - threshold (float64): Threshold to be applied for binarization. If not
        set, the threshold defaults to 0.0.  Default value 0.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Matrix in which to save the output.

*/
func Preprocess_binarize(input *mat.Dense, param *Preprocess_binarizeOptionalParam) (*mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Binarize Data")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  GonumToArmaMat("input", input)
  SetPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Dimension != 0 {
    SetParamInt("dimension", param.Dimension)
    SetPassed("dimension")
  }

  // Detect if the parameter was passed; set if so.
  if param.Threshold != 0 {
    SetParamDouble("threshold", param.Threshold)
    SetPassed("threshold")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("output")

  // Call the mlpack program.
  C.mlpackpreprocess_binarize()

  // Initialize result variable and get output.
  var output_ptr mlpackArma
  output := output_ptr.ArmaToGonumMat("output")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output
}
