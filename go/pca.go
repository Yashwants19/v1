package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_pca
#include <capi/pca.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type PcaOptionalParam struct {
    Copy_all_inputs bool
    Decomposition_method string
    New_dimensionality int
    Scale bool
    Var_to_retain float64
    Verbose bool
}

func InitializePca() *PcaOptionalParam {
  return &PcaOptionalParam{
    Copy_all_inputs: false,
    Decomposition_method: "exact",
    New_dimensionality: 0,
    Scale: false,
    Var_to_retain: 0,
    Verbose: false,
  }
}

/*
  This program performs principal components analysis on the given dataset using
  the exact, randomized, randomized block Krylov, or QUIC SVD method. It will
  transform the data onto its principal components, optionally performing
  dimensionality reduction by ignoring the principal components with the
  smallest eigenvalues.
  
  Use the 'input' parameter to specify the dataset to perform PCA on.  A desired
  new dimensionality can be specified with the 'new_dimensionality' parameter,
  or the desired variance to retain can be specified with the 'var_to_retain'
  parameter.  If desired, the dataset can be scaled before running PCA with the
  'scale' parameter.
  
  Multiple different decomposition techniques can be used.  The method to use
  can be specified with the 'decomposition_method' parameter, and it may take
  the values 'exact', 'randomized', or 'quic'.
  
  For example, to reduce the dimensionality of the matrix data to 5 dimensions
  using randomized SVD for the decomposition, storing the output matrix to
  data_mod, the following command can be used:
  
  param := InitializePca()
  param.New_dimensionality = 5
  param.Decomposition_method = "randomized"
  data_mod := Pca(data, param)


  Input parameters:

   - input (mat.Dense): Input dataset to perform PCA on.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - decomposition_method (string): Method used for the principal
        components analysis: 'exact', 'randomized', 'randomized-block-krylov',
        'quic'.  Default value 'exact'.
   - new_dimensionality (int): Desired dimensionality of output dataset.
        If 0, no dimensionality reduction is performed.  Default value 0.
   - scale (bool): If set, the data will be scaled before running PCA,
        such that the variance of each feature is 1.
   - var_to_retain (float64): Amount of variance to retain; should be
        between 0 and 1.  If 1, all variance is retained.  Overrides -d. 
        Default value 0.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Matrix to save modified dataset to.

*/
func Pca(input *mat.Dense, param *PcaOptionalParam) (*mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Principal Components Analysis")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  GonumToArmaMat("input", input)
  SetPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Decomposition_method != "exact" {
    SetParamString("decomposition_method", param.Decomposition_method)
    SetPassed("decomposition_method")
  }

  // Detect if the parameter was passed; set if so.
  if param.New_dimensionality != 0 {
    SetParamInt("new_dimensionality", param.New_dimensionality)
    SetPassed("new_dimensionality")
  }

  // Detect if the parameter was passed; set if so.
  if param.Scale != false {
    SetParamBool("scale", param.Scale)
    SetPassed("scale")
  }

  // Detect if the parameter was passed; set if so.
  if param.Var_to_retain != 0 {
    SetParamDouble("var_to_retain", param.Var_to_retain)
    SetPassed("var_to_retain")
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
  C.mlpackpca()

  // Initialize result variable and get output.
  var output_ptr mlpackArma
  output := output_ptr.ArmaToGonumMat("output")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output
}
