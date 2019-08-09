package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_mean_shift
#include <capi/mean_shift.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type Mean_shiftOptionalParam struct {
    Copy_all_inputs bool
    Force_convergence bool
    In_place bool
    Labels_only bool
    Max_iterations int
    Radius float64
    Verbose bool
}

func InitializeMean_shift() *Mean_shiftOptionalParam {
  return &Mean_shiftOptionalParam{
    Copy_all_inputs: false,
    Force_convergence: false,
    In_place: false,
    Labels_only: false,
    Max_iterations: 1000,
    Radius: 0,
    Verbose: false,
  }
}

/*
  This program performs mean shift clustering on the given dataset, storing the
  learned cluster assignments either as a column of labels in the input dataset
  or separately.
  
  The input dataset should be specified with the 'input' parameter, and the
  radius used for search can be specified with the 'radius' parameter.  The
  maximum number of iterations before algorithm termination is controlled with
  the 'max_iterations' parameter.
  
  The output labels may be saved with the 'output' output parameter and the
  centroids of each cluster may be saved with the 'centroid' output parameter.
  
  For example, to run mean shift clustering on the dataset data and store the
  centroids to centroids, the following command may be used: 
  
  param := InitializeMean_shift()
  centroids, _ := Mean_shift(data, )


  Input parameters:

   - input (mat.Dense): Input dataset to perform clustering on.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - force_convergence (bool): If specified, the mean shift algorithm will
        continue running regardless of max_iterations until the clusters
        converge.
   - in_place (bool): If specified, a column containing the learned
        cluster assignments will be added to the input dataset file.  In this
        case, --output_file is overridden.  (Do not use with Python.)
   - labels_only (bool): If specified, only the output labels will be
        written to the file specified by --output_file.
   - max_iterations (int): Maximum number of iterations before mean shift
        terminates.  Default value 1000.
   - radius (float64): If the distance between two centroids is less than
        the given radius, one will be removed.  A radius of 0 or less means an
        estimate will be calculated and used for the radius.  Default value 0.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - centroid (mat.Dense): If specified, the centroids of each cluster
        will be written to the given matrix.
   - output (mat.Dense): Matrix to write output labels or labeled data
        to.

*/
func Mean_shift(input *mat.Dense, param *Mean_shiftOptionalParam) (*mat.Dense, *mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Mean Shift Clustering")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  GonumToArmaMat("input", input)
  SetPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Force_convergence != false {
    SetParamBool("force_convergence", param.Force_convergence)
    SetPassed("force_convergence")
  }

  // Detect if the parameter was passed; set if so.
  if param.In_place != false {
    SetParamBool("in_place", param.In_place)
    SetPassed("in_place")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels_only != false {
    SetParamBool("labels_only", param.Labels_only)
    SetPassed("labels_only")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max_iterations != 1000 {
    SetParamInt("max_iterations", param.Max_iterations)
    SetPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Radius != 0 {
    SetParamDouble("radius", param.Radius)
    SetPassed("radius")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("centroid")
  SetPassed("output")

  // Call the mlpack program.
  C.mlpackmean_shift()

  // Initialize result variable and get output.
  var centroid_ptr mlpackArma
  centroid := centroid_ptr.ArmaToGonumMat("centroid")
  var output_ptr mlpackArma
  output := output_ptr.ArmaToGonumMat("output")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return centroid, output
}
