package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_gmm_train
#include <capi/gmm_train.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type Gmm_trainOptionalParam struct {
    Copy_all_inputs bool
    Diagonal_covariance bool
    Input_model *GMM
    Max_iterations int
    No_force_positive bool
    Noise float64
    Percentage float64
    Refined_start bool
    Samplings int
    Seed int
    Tolerance float64
    Trials int
    Verbose bool
}

func InitializeGmm_train() *Gmm_trainOptionalParam {
  return &Gmm_trainOptionalParam{
    Copy_all_inputs: false,
    Diagonal_covariance: false,
    Input_model: nil,
    Max_iterations: 250,
    No_force_positive: false,
    Noise: 0,
    Percentage: 0.02,
    Refined_start: false,
    Samplings: 100,
    Seed: 0,
    Tolerance: 1e-10,
    Trials: 1,
    Verbose: false,
  }
}

type GMM struct {
 mem unsafe.Pointer
}

func (m *GMM) allocGMM(identifier string) {
 m.mem = C.mlpackGetGMMPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *GMM) getGMM(identifier string) {
 m.allocGMM(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setGMM(identifier string, ptr *GMM) {
 C.mlpackSetGMMPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program takes a parametric estimate of a Gaussian mixture model (GMM)
  using the EM algorithm to find the maximum likelihood estimate.  The model may
  be saved and reused by other mlpack GMM tools.
  
  The input data to train on must be specified with the 'input' parameter, and
  the number of Gaussians in the model must be specified with the 'gaussians'
  parameter.  Optionally, many trials with different random initializations may
  be run, and the result with highest log-likelihood on the training data will
  be taken.  The number of trials to run is specified with the 'trials'
  parameter.  By default, only one trial is run.
  
  The tolerance for convergence and maximum number of iterations of the EM
  algorithm are specified with the 'tolerance' and 'max_iterations' parameters,
  respectively.  The GMM may be initialized for training with another model,
  specified with the 'input_model' parameter. Otherwise, the model is
  initialized by running k-means on the data.  The k-means clustering
  initialization can be controlled with the 'refined_start', 'samplings', and
  'percentage' parameters.  If 'refined_start' is specified, then the
  Bradley-Fayyad refined start initialization will be used.  This can often lead
  to better clustering results.
  
  The 'diagonal_covariance' flag will cause the learned covariances to be
  diagonal matrices.  This significantly simplifies the model itself and causes
  training to be faster, but restricts the ability to fit more complex GMMs.
  
  If GMM training fails with an error indicating that a covariance matrix could
  not be inverted, make sure that the 'no_force_positive' parameter is not
  specified.  Alternately, adding a small amount of Gaussian noise (using the
  'noise' parameter) to the entire dataset may help prevent Gaussians with zero
  variance in a particular dimension, which is usually the cause of
  non-invertible covariance matrices.
  
  The 'no_force_positive' parameter, if set, will avoid the checks after each
  iteration of the EM algorithm which ensure that the covariance matrices are
  positive definite.  Specifying the flag can cause faster runtime, but may also
  cause non-positive definite covariance matrices, which will cause the program
  to crash.
  
  As an example, to train a 6-Gaussian GMM on the data in data with a maximum of
  100 iterations of EM and 3 trials, saving the trained GMM to gmm, the
  following command can be used:
  
  param := InitializeGmm_train()
  param.Trials = 3
  gmm := Gmm_train(data, 6, param)
  
  To re-train that GMM on another set of data data2, the following command may
  be used: 
  
  param := InitializeGmm_train()
  param.Input_model = gmm
  new_gmm := Gmm_train(data2, 6, param)


  Input parameters:

   - gaussians (int): Number of Gaussians in the GMM.
   - input (mat.Dense): The training data on which the model will be fit.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - diagonal_covariance (bool): Force the covariance of the Gaussians to
        be diagonal.  This can accelerate training time significantly.
   - input_model (GMM): Initial input GMM model to start training with.
   - max_iterations (int): Maximum number of iterations of EM algorithm
        (passing 0 will run until convergence).  Default value 250.
   - no_force_positive (bool): Do not force the covariance matrices to be
        positive definite.
   - noise (float64): Variance of zero-mean Gaussian noise to add to data.
         Default value 0.
   - percentage (float64): If using --refined_start, specify the
        percentage of the dataset used for each sampling (should be between 0.0
        and 1.0).  Default value 0.02.
   - refined_start (bool): During the initialization, use refined initial
        positions for k-means clustering (Bradley and Fayyad, 1998).
   - samplings (int): If using --refined_start, specify the number of
        samplings used for initial points.  Default value 100.
   - seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - tolerance (float64): Tolerance for convergence of EM.  Default value
        1e-10.
   - trials (int): Number of trials to perform in training GMM.  Default
        value 1.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output_model (GMM): Output for trained GMM model.

*/
func Gmm_train(gaussians int, input *mat.Dense, param *Gmm_trainOptionalParam) (GMM) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Gaussian Mixture Model (GMM) Training")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  SetParamInt("gaussians", gaussians)
  SetPassed("gaussians")

  // Detect if the parameter was passed; set if so.
  GonumToArmaMat("input", input)
  SetPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Diagonal_covariance != false {
    SetParamBool("diagonal_covariance", param.Diagonal_covariance)
    SetPassed("diagonal_covariance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setGMM("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max_iterations != 250 {
    SetParamInt("max_iterations", param.Max_iterations)
    SetPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.No_force_positive != false {
    SetParamBool("no_force_positive", param.No_force_positive)
    SetPassed("no_force_positive")
  }

  // Detect if the parameter was passed; set if so.
  if param.Noise != 0 {
    SetParamDouble("noise", param.Noise)
    SetPassed("noise")
  }

  // Detect if the parameter was passed; set if so.
  if param.Percentage != 0.02 {
    SetParamDouble("percentage", param.Percentage)
    SetPassed("percentage")
  }

  // Detect if the parameter was passed; set if so.
  if param.Refined_start != false {
    SetParamBool("refined_start", param.Refined_start)
    SetPassed("refined_start")
  }

  // Detect if the parameter was passed; set if so.
  if param.Samplings != 100 {
    SetParamInt("samplings", param.Samplings)
    SetPassed("samplings")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    SetParamInt("seed", param.Seed)
    SetPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tolerance != 1e-10 {
    SetParamDouble("tolerance", param.Tolerance)
    SetPassed("tolerance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Trials != 1 {
    SetParamInt("trials", param.Trials)
    SetPassed("trials")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("output_model")

  // Call the mlpack program.
  C.mlpackgmm_train()

  // Initialize result variable and get output.
  var output_model GMM
  output_model.getGMM("output_model")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output_model
}
