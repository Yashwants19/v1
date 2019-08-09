package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_radical
#include <capi/radical.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type RadicalOptionalParam struct {
    Angles int
    Copy_all_inputs bool
    Noise_std_dev float64
    Objective bool
    Replicates int
    Seed int
    Sweeps int
    Verbose bool
}

func InitializeRadical() *RadicalOptionalParam {
  return &RadicalOptionalParam{
    Angles: 150,
    Copy_all_inputs: false,
    Noise_std_dev: 0.175,
    Objective: false,
    Replicates: 30,
    Seed: 0,
    Sweeps: 0,
    Verbose: false,
  }
}

/*
  An implementation of RADICAL, a method for independent component analysis
  (ICA).  Assuming that we have an input matrix X, the goal is to find a square
  unmixing matrix W such that Y = W * X and the dimensions of Y are independent
  components.  If the algorithm is running particularly slowly, try reducing the
  number of replicates.
  
  The input matrix to perform ICA on should be specified with the 'input'
  parameter.  The output matrix Y may be saved with the 'output_ic' output
  parameter, and the output unmixing matrix W may be saved with the
  'output_unmixing' output parameter.
  
  For example, to perform ICA on the matrix X with 40 replicates, saving the
  independent components to ic, the following command may be used: 
  
  param := InitializeRadical()
  param.Replicates = 40
  ic, _ := Radical(X, param)


  Input parameters:

   - input (mat.Dense): Input dataset for ICA.
   - angles (int): Number of angles to consider in brute-force search
        during Radical2D.  Default value 150.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - noise_std_dev (float64): Standard deviation of Gaussian noise. 
        Default value 0.175.
   - objective (bool): If set, an estimate of the final objective function
        is printed.
   - replicates (int): Number of Gaussian-perturbed replicates to use (per
        point) in Radical2D.  Default value 30.
   - seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - sweeps (int): Number of sweeps; each sweep calls Radical2D once for
        each pair of dimensions.  Default value 0.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output_ic (mat.Dense): Matrix to save independent components to.
   - output_unmixing (mat.Dense): Matrix to save unmixing matrix to.

*/
func Radical(input *mat.Dense, param *RadicalOptionalParam) (*mat.Dense, *mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("RADICAL")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  GonumToArmaMat("input", input)
  SetPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Angles != 150 {
    SetParamInt("angles", param.Angles)
    SetPassed("angles")
  }

  // Detect if the parameter was passed; set if so.
  if param.Noise_std_dev != 0.175 {
    SetParamDouble("noise_std_dev", param.Noise_std_dev)
    SetPassed("noise_std_dev")
  }

  // Detect if the parameter was passed; set if so.
  if param.Objective != false {
    SetParamBool("objective", param.Objective)
    SetPassed("objective")
  }

  // Detect if the parameter was passed; set if so.
  if param.Replicates != 30 {
    SetParamInt("replicates", param.Replicates)
    SetPassed("replicates")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    SetParamInt("seed", param.Seed)
    SetPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Sweeps != 0 {
    SetParamInt("sweeps", param.Sweeps)
    SetPassed("sweeps")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("output_ic")
  SetPassed("output_unmixing")

  // Call the mlpack program.
  C.mlpackradical()

  // Initialize result variable and get output.
  var output_ic_ptr mlpackArma
  output_ic := output_ic_ptr.ArmaToGonumMat("output_ic")
  var output_unmixing_ptr mlpackArma
  output_unmixing := output_unmixing_ptr.ArmaToGonumMat("output_unmixing")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output_ic, output_unmixing
}
