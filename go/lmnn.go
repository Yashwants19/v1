package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_lmnn
#include <capi/lmnn.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type LmnnOptionalParam struct {
    Batch_size int
    Center bool
    Copy_all_inputs bool
    Distance *mat.Dense
    K int
    Labels *mat.VecDense
    Linear_scan bool
    Max_iterations int
    Normalize bool
    Optimizer string
    Passes int
    Print_accuracy bool
    Range int
    Rank int
    Regularization float64
    Seed int
    Step_size float64
    Tolerance float64
    Verbose bool
}

func InitializeLmnn() *LmnnOptionalParam {
  return &LmnnOptionalParam{
    Batch_size: 50,
    Center: false,
    Copy_all_inputs: false,
    Distance: nil,
    K: 1,
    Labels: nil,
    Linear_scan: false,
    Max_iterations: 100000,
    Normalize: false,
    Optimizer: "amsgrad",
    Passes: 50,
    Print_accuracy: false,
    Range: 1,
    Rank: 0,
    Regularization: 0.5,
    Seed: 0,
    Step_size: 0.01,
    Tolerance: 1e-07,
    Verbose: false,
  }
}

/*
  This program implements Large Margin Nearest Neighbors, a distance learning
  technique.  The method seeks to improve k-nearest-neighbor classification on a
  dataset.  The method employes the strategy of reducing distance between
  similar labeled data points (a.k.a target neighbors) and increasing distance
  between differently labeled points (a.k.a impostors) using standard
  optimization techniques over the gradient of the distance between data points.
  
  To work, this algorithm needs labeled data.  It can be given as the last row
  of the input dataset (specified with 'input'), or alternatively as a separate
  matrix (specified with 'labels').  Additionally, a starting point for
  optimization (specified with 'distance'can be given, having (r x d)
  dimensionality.  Here r should satisfy 1 <= r <= d, Consequently a Low-Rank
  matrix will be optimized. Alternatively, Low-Rank distance can be learned by
  specifying the 'rank'parameter (A Low-Rank matrix with uniformly distributed
  values will be used as initial learning point). 
  
  The program also requires number of targets neighbors to work with ( specified
  with 'k'), A regularization parameter can also be passed, It acts as a trade
  of between the pulling and pushing terms (specified with 'regularization'), In
  addition, this implementation of LMNN includes a parameter to decide the
  interval after which impostors must be re-calculated (specified with 'range').
  
  Output can either be the learned distance matrix (specified with 'output'), or
  the transformed dataset  (specified with 'transformed_data'), or both.
  Additionally mean-centered dataset (specified with 'centered_data') can be
  accessed given mean-centering (specified with 'center') is performed on the
  dataset. Accuracy on initial dataset and final transformed dataset can be
  printed by specifying the 'print_accuracy'parameter. 
  
  This implementation of LMNN uses AdaGrad, BigBatch_SGD, stochastic gradient
  descent, mini-batch stochastic gradient descent, or the L_BFGS optimizer. 
  
  AdaGrad, specified by the value 'adagrad' for the parameter 'optimizer', uses
  maximum of past squared gradients. It primarily on six parameters: the step
  size (specified with 'step_size'), the batch size (specified with
  'batch_size'), the maximum number of passes (specified with 'passes').
  Inaddition, a normalized starting point can be used by specifying the
  'normalize' parameter. 
  
  BigBatch_SGD, specified by the value 'bbsgd' for the parameter 'optimizer',
  depends primarily on four parameters: the step size (specified with
  'step_size'), the batch size (specified with 'batch_size'), the maximum number
  of passes (specified with 'passes').  In addition, a normalized starting point
  can be used by specifying the 'normalize' parameter. 
  
  Stochastic gradient descent, specified by the value 'sgd' for the parameter
  'optimizer', depends primarily on three parameters: the step size (specified
  with 'step_size'), the batch size (specified with 'batch_size'), and the
  maximum number of passes (specified with 'passes').  In addition, a normalized
  starting point can be used by specifying the 'normalize' parameter.
  Furthermore, mean-centering can be performed on the dataset by specifying the
  'center'parameter. 
  
  The L-BFGS optimizer, specified by the value 'lbfgs' for the parameter
  'optimizer', uses a back-tracking line search algorithm to minimize a
  function.  The following parameters are used by L-BFGS: 'max_iterations',
  'tolerance'(the optimization is terminated when the gradient norm is below
  this value).  For more details on the L-BFGS optimizer, consult either the
  mlpack L-BFGS documentation (in lbfgs.hpp) or the vast set of published
  literature on L-BFGS.  In addition, a normalized starting point can be used by
  specifying the 'normalize' parameter.
  
  By default, the AMSGrad optimizer is used.
  
  Example - Let's say we want to learn distance on iris dataset with number of
  targets as 3 using BigBatch_SGD optimizer. A simple call for the same will
  look like: 
  
  param := InitializeMlpack_lmnn()
  param.Labels = iris_labels
  param.K = 3
  param.Optimizer = "bbsgd"
  _, output, _ := Mlpack_lmnn(iris, param)
  
  An another program call making use of range & regularization parameter with
  dataset having labels as last column can be made as: 
  
  param := InitializeMlpack_lmnn()
  param.K = 5
  param.Range = 10
  param.Regularization = 0.4
  _, output, _ := Mlpack_lmnn(letter_recognition, param)


  Input parameters:

   - input (mat.Dense): Input dataset to run LMNN on.
   - batch_size (int): Batch size for mini-batch SGD.  Default value 50.
   - center (bool): Perform mean-centering on the dataset. It is useful
        when the centroid of the data is far from the origin.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - distance (mat.Dense): Initial distance matrix to be used as starting
        point
   - k (int): Number of target neighbors to use for each datapoint. 
        Default value 1.
   - labels (mat.VecDense): Labels for input dataset.
   - linear_scan (bool): Don't shuffle the order in which data points are
        visited for SGD or mini-batch SGD.
   - max_iterations (int): Maximum number of iterations for L-BFGS (0
        indicates no limit).  Default value 100000.
   - normalize (bool): Use a normalized starting point for optimization.
        Itis useful for when points are far apart, or when SGD is returning
        NaN.
   - optimizer (string): Optimizer to use; 'amsgrad', 'bbsgd', 'sgd', or
        'lbfgs'.  Default value 'amsgrad'.
   - passes (int): Maximum number of full passes over dataset for AMSGrad,
        BB_SGD and SGD.  Default value 50.
   - print_accuracy (bool): Print accuracies on initial and transformed
        dataset
   - range (int): Number of iterations after which impostors needs to be
        recalculated  Default value 1.
   - rank (int): Rank of distance matrix to be optimized.   Default value
        0.
   - regularization (float64): Regularization for LMNN objective function 
         Default value 0.5.
   - seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - step_size (float64): Step size for AMSGrad, BB_SGD and SGD (alpha). 
        Default value 0.01.
   - tolerance (float64): Maximum tolerance for termination of AMSGrad,
        BB_SGD, SGD or L-BFGS.  Default value 1e-07.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - centered_data (mat.Dense): Output matrix for mean-centered dataset.
   - output (mat.Dense): Output matrix for learned distance matrix.
   - transformed_data (mat.Dense): Output matrix for transformed dataset.

*/
func Lmnn(input *mat.Dense, param *LmnnOptionalParam) (*mat.Dense, *mat.Dense, *mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Large Margin Nearest Neighbors (LMNN)")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  GonumToArmaMat("input", input)
  SetPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Batch_size != 50 {
    SetParamInt("batch_size", param.Batch_size)
    SetPassed("batch_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Center != false {
    SetParamBool("center", param.Center)
    SetPassed("center")
  }

  // Detect if the parameter was passed; set if so.
  if param.Distance != nil {
    GonumToArmaMat("distance", param.Distance)
    SetPassed("distance")
  }

  // Detect if the parameter was passed; set if so.
  if param.K != 1 {
    SetParamInt("k", param.K)
    SetPassed("k")
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
  if param.Max_iterations != 100000 {
    SetParamInt("max_iterations", param.Max_iterations)
    SetPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Normalize != false {
    SetParamBool("normalize", param.Normalize)
    SetPassed("normalize")
  }

  // Detect if the parameter was passed; set if so.
  if param.Optimizer != "amsgrad" {
    SetParamString("optimizer", param.Optimizer)
    SetPassed("optimizer")
  }

  // Detect if the parameter was passed; set if so.
  if param.Passes != 50 {
    SetParamInt("passes", param.Passes)
    SetPassed("passes")
  }

  // Detect if the parameter was passed; set if so.
  if param.Print_accuracy != false {
    SetParamBool("print_accuracy", param.Print_accuracy)
    SetPassed("print_accuracy")
  }

  // Detect if the parameter was passed; set if so.
  if param.Range != 1 {
    SetParamInt("range", param.Range)
    SetPassed("range")
  }

  // Detect if the parameter was passed; set if so.
  if param.Rank != 0 {
    SetParamInt("rank", param.Rank)
    SetPassed("rank")
  }

  // Detect if the parameter was passed; set if so.
  if param.Regularization != 0.5 {
    SetParamDouble("regularization", param.Regularization)
    SetPassed("regularization")
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

  // Mark all output options as passed.
  SetPassed("centered_data")
  SetPassed("output")
  SetPassed("transformed_data")

  // Call the mlpack program.
  C.mlpacklmnn()

  // Initialize result variable and get output.
  var centered_data_ptr mlpackArma
  centered_data := centered_data_ptr.ArmaToGonumMat("centered_data")
  var output_ptr mlpackArma
  output := output_ptr.ArmaToGonumMat("output")
  var transformed_data_ptr mlpackArma
  transformed_data := transformed_data_ptr.ArmaToGonumMat("transformed_data")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return centered_data, output, transformed_data
}
