package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_logistic_regression
#include <capi/logistic_regression.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type Logistic_regressionOptionalParam struct {
    Batch_size int
    Copy_all_inputs bool
    Decision_boundary float64
    Input_model *LogisticRegression
    Labels *mat.VecDense
    Lambda float64
    Max_iterations int
    Optimizer string
    Step_size float64
    Test *mat.Dense
    Tolerance float64
    Training *mat.Dense
    Verbose bool
}

func InitializeLogistic_regression() *Logistic_regressionOptionalParam {
  return &Logistic_regressionOptionalParam{
    Batch_size: 64,
    Copy_all_inputs: false,
    Decision_boundary: 0.5,
    Input_model: nil,
    Labels: nil,
    Lambda: 0,
    Max_iterations: 10000,
    Optimizer: "lbfgs",
    Step_size: 0.01,
    Test: nil,
    Tolerance: 1e-10,
    Training: nil,
    Verbose: false,
  }
}

type LogisticRegression struct {
 mem unsafe.Pointer
}

func (m *LogisticRegression) allocLogisticRegression(identifier string) {
 m.mem = C.mlpackGetLogisticRegressionPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *LogisticRegression) getLogisticRegression(identifier string) {
 m.allocLogisticRegression(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setLogisticRegression(identifier string, ptr *LogisticRegression) {
 C.mlpackSetLogisticRegressionPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  An implementation of L2-regularized logistic regression using either the
  L-BFGS optimizer or SGD (stochastic gradient descent).  This solves the
  regression problem
  
    y = (1 / 1 + e^-(X * b))
  
  where y takes values 0 or 1.
  
  This program allows loading a logistic regression model (via the 'input_model'
  parameter) or training a logistic regression model given training data
  (specified with the 'training' parameter), or both those things at once.  In
  addition, this program allows classification on a test dataset (specified with
  the 'test' parameter) and the classification results may be saved with the
  'predictions' output parameter. The trained logistic regression model may be
  saved using the 'output_model' output parameter.
  
  The training data, if specified, may have class labels as its last dimension. 
  Alternately, the 'labels' parameter may be used to specify a separate matrix
  of labels.
  
  When a model is being trained, there are many options.  L2 regularization (to
  prevent overfitting) can be specified with the 'lambda' option, and the
  optimizer used to train the model can be specified with the 'optimizer'
  parameter.  Available options are 'sgd' (stochastic gradient descent) and
  'lbfgs' (the L-BFGS optimizer).  There are also various parameters for the
  optimizer; the 'max_iterations' parameter specifies the maximum number of
  allowed iterations, and the 'tolerance' parameter specifies the tolerance for
  convergence.  For the SGD optimizer, the 'step_size' parameter controls the
  step size taken at each iteration by the optimizer.  The batch size for SGD is
  controlled with the 'batch_size' parameter. If the objective function for your
  data is oscillating between Inf and 0, the step size is probably too large. 
  There are more parameters for the optimizers, but the C++ interface must be
  used to access these.
  
  For SGD, an iteration refers to a single point. So to take a single pass over
  the dataset with SGD, 'max_iterations' should be set to the number of points
  in the dataset.
  
  Optionally, the model can be used to predict the responses for another matrix
  of data points, if 'test' is specified.  The 'test' parameter can be specified
  without the 'training' parameter, so long as an existing logistic regression
  model is given with the 'input_model' parameter.  The output predictions from
  the logistic regression model may be saved with the 'predictions' parameter.
  
  Note : The following parameters are deprecated and will be removed in mlpack
  4: 'output', 'output_probabilities'
  Use 'predictions' instead of 'output'
  Use 'probabilities' instead of 'output_probabilities'
  
  This implementation of logistic regression does not support the general
  multi-class case but instead only the two-class case.  Any labels must be
  either 0 or 1.  For more classes, see the softmax_regression program.
  
  As an example, to train a logistic regression model on the data 'data' with
  labels 'labels' with L2 regularization of 0.1, saving the model to 'lr_model',
  the following command may be used:
  
  param := InitializeLogistic_regression()
  param.Training = data
  param.Labels = labels
  param.Lambda = 0.1
  _, lr_model, _, _, _ := Logistic_regression(param)
  
  Then, to use that model to predict classes for the dataset 'test', storing the
  output predictions in 'predictions', the following command may be used: 
  
  param := InitializeLogistic_regression()
  param.Input_model = lr_model
  param.Test = test
  predictions, _, _, _, _ := Logistic_regression(param)


  Input parameters:

   - batch_size (int): Batch size for SGD.  Default value 64.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - decision_boundary (float64): Decision boundary for prediction; if the
        logistic function for a point is less than the boundary, the class is
        taken to be 0; otherwise, the class is 1.  Default value 0.5.
   - input_model (LogisticRegression): Existing model (parameters).
   - labels (mat.VecDense): A matrix containing labels (0 or 1) for the
        points in the training set (y).
   - lambda (float64): L2-regularization parameter for training.  Default
        value 0.
   - max_iterations (int): Maximum iterations for optimizer (0 indicates
        no limit).  Default value 10000.
   - optimizer (string): Optimizer to use for training ('lbfgs' or 'sgd').
         Default value 'lbfgs'.
   - step_size (float64): Step size for SGD optimizer.  Default value
        0.01.
   - test (mat.Dense): Matrix containing test dataset.
   - tolerance (float64): Convergence tolerance for optimizer.  Default
        value 1e-10.
   - training (mat.Dense): A matrix containing the training set (the
        matrix of predictors, X).
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.VecDense): If test data is specified, this matrix is
        where the predictions for the test set will be saved.
   - output_model (LogisticRegression): Output for trained logistic
        regression model.
   - output_probabilities (mat.Dense): If test data is specified, this
        matrix is where the class probabilities for the test set will be saved.
   - predictions (mat.VecDense): If test data is specified, this matrix is
        where the predictions for the test set will be saved.
   - probabilities (mat.Dense): If test data is specified, this matrix is
        where the class probabilities for the test set will be saved.

*/
func Logistic_regression(param *Logistic_regressionOptionalParam) (*mat.VecDense, LogisticRegression, *mat.Dense, *mat.VecDense, *mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("L2-regularized Logistic Regression and Prediction")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Batch_size != 64 {
    SetParamInt("batch_size", param.Batch_size)
    SetPassed("batch_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Decision_boundary != 0.5 {
    SetParamDouble("decision_boundary", param.Decision_boundary)
    SetPassed("decision_boundary")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setLogisticRegression("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    GonumToArmaUrow("labels", param.Labels)
    SetPassed("labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda != 0 {
    SetParamDouble("lambda", param.Lambda)
    SetPassed("lambda")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max_iterations != 10000 {
    SetParamInt("max_iterations", param.Max_iterations)
    SetPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Optimizer != "lbfgs" {
    SetParamString("optimizer", param.Optimizer)
    SetPassed("optimizer")
  }

  // Detect if the parameter was passed; set if so.
  if param.Step_size != 0.01 {
    SetParamDouble("step_size", param.Step_size)
    SetPassed("step_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    GonumToArmaMat("test", param.Test)
    SetPassed("test")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tolerance != 1e-10 {
    SetParamDouble("tolerance", param.Tolerance)
    SetPassed("tolerance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training != nil {
    GonumToArmaMat("training", param.Training)
    SetPassed("training")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("output")
  SetPassed("output_model")
  SetPassed("output_probabilities")
  SetPassed("predictions")
  SetPassed("probabilities")

  // Call the mlpack program.
  C.mlpacklogistic_regression()

  // Initialize result variable and get output.
  var output_ptr mlpackArma
  output := output_ptr.ArmaToGonumUrow("output")
  var output_model LogisticRegression
  output_model.getLogisticRegression("output_model")
  var output_probabilities_ptr mlpackArma
  output_probabilities := output_probabilities_ptr.ArmaToGonumMat("output_probabilities")
  var predictions_ptr mlpackArma
  predictions := predictions_ptr.ArmaToGonumUrow("predictions")
  var probabilities_ptr mlpackArma
  probabilities := probabilities_ptr.ArmaToGonumMat("probabilities")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output, output_model, output_probabilities, predictions, probabilities
}
