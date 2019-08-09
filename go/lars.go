package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_lars
#include <capi/lars.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type LarsOptionalParam struct {
    Copy_all_inputs bool
    Input *mat.Dense
    Input_model *LARS
    Lambda1 float64
    Lambda2 float64
    Responses *mat.Dense
    Test *mat.Dense
    Use_cholesky bool
    Verbose bool
}

func InitializeLars() *LarsOptionalParam {
  return &LarsOptionalParam{
    Copy_all_inputs: false,
    Input: nil,
    Input_model: nil,
    Lambda1: 0,
    Lambda2: 0,
    Responses: nil,
    Test: nil,
    Use_cholesky: false,
    Verbose: false,
  }
}

type LARS struct {
 mem unsafe.Pointer
}

func (m *LARS) allocLARS(identifier string) {
 m.mem = C.mlpackGetLARSPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *LARS) getLARS(identifier string) {
 m.allocLARS(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setLARS(identifier string, ptr *LARS) {
 C.mlpackSetLARSPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  An implementation of LARS: Least Angle Regression (Stagewise/laSso).  This is
  a stage-wise homotopy-based algorithm for L1-regularized linear regression
  (LASSO) and L1+L2-regularized linear regression (Elastic Net).
  
  This program is able to train a LARS/LASSO/Elastic Net model or load a model
  from file, output regression predictions for a test set, and save the trained
  model to a file.  The LARS algorithm is described in more detail below:
  
  Let X be a matrix where each row is a point and each column is a dimension,
  and let y be a vector of targets.
  
  The Elastic Net problem is to solve
  
    min_beta 0.5 || X * beta - y ||_2^2 + lambda_1 ||beta||_1 +
        0.5 lambda_2 ||beta||_2^2
  
  If lambda1 > 0 and lambda2 = 0, the problem is the LASSO.
  If lambda1 > 0 and lambda2 > 0, the problem is the Elastic Net.
  If lambda1 = 0 and lambda2 > 0, the problem is ridge regression.
  If lambda1 = 0 and lambda2 = 0, the problem is unregularized linear
  regression.
  
  For efficiency reasons, it is not recommended to use this algorithm with
  'lambda1' = 0.  In that case, use the 'linear_regression' program, which
  implements both unregularized linear regression and ridge regression.
  
  To train a LARS/LASSO/Elastic Net model, the 'input' and 'responses'
  parameters must be given.  The 'lambda1', 'lambda2', and 'use_cholesky'
  parameters control the training options.  A trained model can be saved with
  the 'output_model'.  If no training is desired at all, a model can be passed
  via the 'input_model' parameter.
  
  The program can also provide predictions for test data using either the
  trained model or the given input model.  Test points can be specified with the
  'test' parameter.  Predicted responses to the test points can be saved with
  the 'output_predictions' output parameter.
  
  For example, the following command trains a model on the data data and
  responses responses with lambda1 set to 0.4 and lambda2 set to 0 (so, LASSO is
  being solved), and then the model is saved to lasso_model:
  
  param := InitializeLars()
  param.Input = data
  param.Responses = responses
  param.Lambda1 = 0.4
  param.Lambda2 = 0
  lasso_model, _ := Lars(param)
  
  The following command uses the lasso_model to provide predicted responses for
  the data test and save those responses to test_predictions: 
  
  param := InitializeLars()
  param.Input_model = lasso_model
  param.Test = test
  _, test_predictions := Lars(param)


  Input parameters:

   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - input (mat.Dense): Matrix of covariates (X).
   - input_model (LARS): Trained LARS model to use.
   - lambda1 (float64): Regularization parameter for l1-norm penalty. 
        Default value 0.
   - lambda2 (float64): Regularization parameter for l2-norm penalty. 
        Default value 0.
   - responses (mat.Dense): Matrix of responses/observations (y).
   - test (mat.Dense): Matrix containing points to regress on (test
        points).
   - use_cholesky (bool): Use Cholesky decomposition during computation
        rather than explicitly computing the full Gram matrix.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output_model (LARS): Output LARS model.
   - output_predictions (mat.Dense): If --test_file is specified, this
        file is where the predicted responses will be saved.

*/
func Lars(param *LarsOptionalParam) (LARS, *mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("LARS")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input != nil {
    GonumToArmaMat("input", param.Input)
    SetPassed("input")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setLARS("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda1 != 0 {
    SetParamDouble("lambda1", param.Lambda1)
    SetPassed("lambda1")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda2 != 0 {
    SetParamDouble("lambda2", param.Lambda2)
    SetPassed("lambda2")
  }

  // Detect if the parameter was passed; set if so.
  if param.Responses != nil {
    GonumToArmaMat("responses", param.Responses)
    SetPassed("responses")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    GonumToArmaMat("test", param.Test)
    SetPassed("test")
  }

  // Detect if the parameter was passed; set if so.
  if param.Use_cholesky != false {
    SetParamBool("use_cholesky", param.Use_cholesky)
    SetPassed("use_cholesky")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("output_model")
  SetPassed("output_predictions")

  // Call the mlpack program.
  C.mlpacklars()

  // Initialize result variable and get output.
  var output_model LARS
  output_model.getLARS("output_model")
  var output_predictions_ptr mlpackArma
  output_predictions := output_predictions_ptr.ArmaToGonumMat("output_predictions")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output_model, output_predictions
}
