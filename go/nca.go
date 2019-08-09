package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_nca
#include <capi/nca.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type NcaOptionalParam struct {
    Armijo_constant float64
    Batch_size int
    Copy_all_inputs bool
    Labels *mat.VecDense
    Linear_scan bool
    Max_iterations int
    Max_line_search_trials int
    Max_step float64
    Min_step float64
    Normalize bool
    Num_basis int
    Optimizer string
    Seed int
    Step_size float64
    Tolerance float64
    Verbose bool
    Wolfe float64
}

func InitializeNca() *NcaOptionalParam {
  return &NcaOptionalParam{
    Armijo_constant: 0.0001,
    Batch_size: 50,
    Copy_all_inputs: false,
    Labels: nil,
    Linear_scan: false,
    Max_iterations: 500000,
    Max_line_search_trials: 50,
    Max_step: 1e+20,
    Min_step: 1e-20,
    Normalize: false,
    Num_basis: 5,
    Optimizer: "sgd",
    Seed: 0,
    Step_size: 0.01,
    Tolerance: 1e-07,
    Verbose: false,
    Wolfe: 0.9,
  }
}

/*
  This program implements Neighborhood Components Analysis, both a linear
  dimensionality reduction technique and a distance learning technique.  The
  method seeks to improve k-nearest-neighbor classification on a dataset by
  scaling the dimensions.  The method is nonparametric, and does not require a
  value of k.  It works by using stochastic ("soft") neighbor assignments and
  using optimization techniques over the gradient of the accuracy of the
  neighbor assignments.
  
  To work, this algorithm needs labeled data.  It can be given as the last row
  of the input dataset (specified with 'input'), or alternatively as a separate
  matrix (specified with 'labels').
  
  This implementation of NCA uses stochastic gradient descent, mini-batch
  stochastic gradient descent, or the L_BFGS optimizer.  These optimizers do not
  guarantee global convergence for a nonconvex objective function (NCA's
  objective function is nonconvex), so the final results could depend on the
  random seed or other optimizer parameters.
  
  Stochastic gradient descent, specified by the value 'sgd' for the parameter
  'optimizer', depends primarily on three parameters: the step size (specified
  with 'step_size'), the batch size (specified with 'batch_size'), and the
  maximum number of iterations (specified with 'max_iterations').  In addition,
  a normalized starting point can be used by specifying the 'normalize'
  parameter, which is necessary if many warnings of the form 'Denominator of p_i
  is 0!' are given.  Tuning the step size can be a tedious affair.  In general,
  the step size is too large if the objective is not mostly uniformly
  decreasing, or if zero-valued denominator warnings are being issued.  The step
  size is too small if the objective is changing very slowly.  Setting the
  termination condition can be done easily once a good step size parameter is
  found; either increase the maximum iterations to a large number and allow SGD
  to find a minimum, or set the maximum iterations to 0 (allowing infinite
  iterations) and set the tolerance (specified by 'tolerance') to define the
  maximum allowed difference between objectives for SGD to terminate.  Be
  careful---setting the tolerance instead of the maximum iterations can take a
  very long time and may actually never converge due to the properties of the
  SGD optimizer. Note that a single iteration of SGD refers to a single point,
  so to take a single pass over the dataset, set the value of the
  'max_iterations' parameter equal to the number of points in the dataset.
  
  The L-BFGS optimizer, specified by the value 'lbfgs' for the parameter
  'optimizer', uses a back-tracking line search algorithm to minimize a
  function.  The following parameters are used by L-BFGS: 'num_basis' (specifies
  the number of memory points used by L-BFGS), 'max_iterations',
  'armijo_constant', 'wolfe', 'tolerance' (the optimization is terminated when
  the gradient norm is below this value), 'max_line_search_trials', 'min_step',
  and 'max_step' (which both refer to the line search routine).  For more
  details on the L-BFGS optimizer, consult either the mlpack L-BFGS
  documentation (in lbfgs.hpp) or the vast set of published literature on
  L-BFGS.
  
  By default, the SGD optimizer is used.


  Input parameters:

   - input (mat.Dense): Input dataset to run NCA on.
   - armijo_constant (float64): Armijo constant for L-BFGS.  Default value
        0.0001.
   - batch_size (int): Batch size for mini-batch SGD.  Default value 50.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - labels (mat.VecDense): Labels for input dataset.
   - linear_scan (bool): Don't shuffle the order in which data points are
        visited for SGD or mini-batch SGD.
   - max_iterations (int): Maximum number of iterations for SGD or L-BFGS
        (0 indicates no limit).  Default value 500000.
   - max_line_search_trials (int): Maximum number of line search trials
        for L-BFGS.  Default value 50.
   - max_step (float64): Maximum step of line search for L-BFGS.  Default
        value 1e+20.
   - min_step (float64): Minimum step of line search for L-BFGS.  Default
        value 1e-20.
   - normalize (bool): Use a normalized starting point for optimization.
        This is useful for when points are far apart, or when SGD is returning
        NaN.
   - num_basis (int): Number of memory points to be stored for L-BFGS. 
        Default value 5.
   - optimizer (string): Optimizer to use; 'sgd' or 'lbfgs'.  Default
        value 'sgd'.
   - seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - step_size (float64): Step size for stochastic gradient descent
        (alpha).  Default value 0.01.
   - tolerance (float64): Maximum tolerance for termination of SGD or
        L-BFGS.  Default value 1e-07.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.
   - wolfe (float64): Wolfe condition parameter for L-BFGS.  Default value
        0.9.

  Output parameters:

   - output (mat.Dense): Output matrix for learned distance matrix.

*/
func Nca(input *mat.Dense, param *NcaOptionalParam) (*mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Neighborhood Components Analysis (NCA)")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  GonumToArmaMat("input", input)
  SetPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Armijo_constant != 0.0001 {
    SetParamDouble("armijo_constant", param.Armijo_constant)
    SetPassed("armijo_constant")
  }

  // Detect if the parameter was passed; set if so.
  if param.Batch_size != 50 {
    SetParamInt("batch_size", param.Batch_size)
    SetPassed("batch_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    GonumToArmaUrow("labels", param.Labels)
    SetPassed("labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Linear_scan != false {
    SetParamBool("linear_scan", param.Linear_scan)
    SetPassed("linear_scan")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max_iterations != 500000 {
    SetParamInt("max_iterations", param.Max_iterations)
    SetPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max_line_search_trials != 50 {
    SetParamInt("max_line_search_trials", param.Max_line_search_trials)
    SetPassed("max_line_search_trials")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max_step != 1e+20 {
    SetParamDouble("max_step", param.Max_step)
    SetPassed("max_step")
  }

  // Detect if the parameter was passed; set if so.
  if param.Min_step != 1e-20 {
    SetParamDouble("min_step", param.Min_step)
    SetPassed("min_step")
  }

  // Detect if the parameter was passed; set if so.
  if param.Normalize != false {
    SetParamBool("normalize", param.Normalize)
    SetPassed("normalize")
  }

  // Detect if the parameter was passed; set if so.
  if param.Num_basis != 5 {
    SetParamInt("num_basis", param.Num_basis)
    SetPassed("num_basis")
  }

  // Detect if the parameter was passed; set if so.
  if param.Optimizer != "sgd" {
    SetParamString("optimizer", param.Optimizer)
    SetPassed("optimizer")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    SetParamInt("seed", param.Seed)
    SetPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Step_size != 0.01 {
    SetParamDouble("step_size", param.Step_size)
    SetPassed("step_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tolerance != 1e-07 {
    SetParamDouble("tolerance", param.Tolerance)
    SetPassed("tolerance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Detect if the parameter was passed; set if so.
  if param.Wolfe != 0.9 {
    SetParamDouble("wolfe", param.Wolfe)
    SetPassed("wolfe")
  }

  // Mark all output options as passed.
  SetPassed("output")

  // Call the mlpack program.
  C.mlpacknca()

  // Initialize result variable and get output.
  var output_ptr mlpackArma
  output := output_ptr.ArmaToGonumMat("output")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output
}
