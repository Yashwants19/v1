package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_linear_regression
#include <capi/linear_regression.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type Linear_regressionOptionalParam struct {
    Copy_all_inputs bool
    Input_model *LinearRegression
    Lambda float64
    Test *mat.Dense
    Training *mat.Dense
    Training_responses *mat.VecDense
    Verbose bool
}

func InitializeLinear_regression() *Linear_regressionOptionalParam {
  return &Linear_regressionOptionalParam{
    Copy_all_inputs: false,
    Input_model: nil,
    Lambda: 0,
    Test: nil,
    Training: nil,
    Training_responses: nil,
    Verbose: false,
  }
}

type LinearRegression struct {
 mem unsafe.Pointer
}

func (m *LinearRegression) allocLinearRegression(identifier string) {
 m.mem = C.mlpackGetLinearRegressionPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *LinearRegression) getLinearRegression(identifier string) {
 m.allocLinearRegression(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setLinearRegression(identifier string, ptr *LinearRegression) {
 C.mlpackSetLinearRegressionPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  An implementation of simple linear regression and simple ridge regression
  using ordinary least squares. This solves the problem
  
    y = X * b + e
  
  where X (specified by 'training') and y (specified either as the last column
  of the input matrix 'training' or via the 'training_responses' parameter) are
  known and b is the desired variable.  If the covariance matrix (X'X) is not
  invertible, or if the solution is overdetermined, then specify a Tikhonov
  regularization constant (with 'lambda') greater than 0, which will regularize
  the covariance matrix to make it invertible.  The calculated b may be saved
  with the 'output_predictions' output parameter.
  
  Optionally, the calculated value of b is used to predict the responses for
  another matrix X' (specified by the 'test' parameter):
  
     y' = X' * b
  
  and the predicted responses y' may be saved with the 'output_predictions'
  output parameter.  This type of regression is related to least-angle
  regression, which mlpack implements as the 'lars' program.
  
  For example, to run a linear regression on the dataset X with responses y,
  saving the trained model to lr_model, the following command could be used:
  
  param := InitializeLinear_regression()
  param.Training = X
  param.Training_responses = y
  lr_model, _ := Linear_regression(param)
  
  Then, to use lr_model to predict responses for a test set X_test, saving the
  predictions to X_test_responses, the following command could be used:
  
  param := InitializeLinear_regression()
  param.Input_model = lr_model
  param.Test = X_test
  _, X_test_responses := Linear_regression(param)


  Input parameters:

   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - input_model (LinearRegression): Existing LinearRegression model to
        use.
   - lambda (float64): Tikhonov regularization for ridge regression.  If
        0, the method reduces to linear regression.  Default value 0.
   - test (mat.Dense): Matrix containing X' (test regressors).
   - training (mat.Dense): Matrix containing training set X (regressors).
   - training_responses (mat.VecDense): Optional vector containing y
        (responses). If not given, the responses are assumed to be the last row
        of the input file.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output_model (LinearRegression): Output LinearRegression model.
   - output_predictions (mat.VecDense): If --test_file is specified, this
        matrix is where the predicted responses will be saved.

*/
func Linear_regression(param *Linear_regressionOptionalParam) (LinearRegression, *mat.VecDense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Simple Linear Regression and Prediction")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setLinearRegression("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda != 0 {
    SetParamDouble("lambda", param.Lambda)
    SetPassed("lambda")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    GonumToArmaMat("test", param.Test)
    SetPassed("test")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training != nil {
    GonumToArmaMat("training", param.Training)
    SetPassed("training")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training_responses != nil {
    GonumToArmaRow("training_responses", param.Training_responses)
    SetPassed("training_responses")
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
  C.mlpacklinear_regression()

  // Initialize result variable and get output.
  var output_model LinearRegression
  output_model.getLinearRegression("output_model")
  var output_predictions_ptr mlpackArma
  output_predictions := output_predictions_ptr.ArmaToGonumRow("output_predictions")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output_model, output_predictions
}
