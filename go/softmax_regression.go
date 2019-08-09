package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_softmax_regression
#include <capi/softmax_regression.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type Softmax_regressionOptionalParam struct {
    Copy_all_inputs bool
    Input_model *SoftmaxRegression
    Labels *mat.VecDense
    Lambda float64
    Max_iterations int
    No_intercept bool
    Number_of_classes int
    Test *mat.Dense
    Test_labels *mat.VecDense
    Training *mat.Dense
    Verbose bool
}

func InitializeSoftmax_regression() *Softmax_regressionOptionalParam {
  return &Softmax_regressionOptionalParam{
    Copy_all_inputs: false,
    Input_model: nil,
    Labels: nil,
    Lambda: 0.0001,
    Max_iterations: 400,
    No_intercept: false,
    Number_of_classes: 0,
    Test: nil,
    Test_labels: nil,
    Training: nil,
    Verbose: false,
  }
}

type SoftmaxRegression struct {
 mem unsafe.Pointer
}

func (m *SoftmaxRegression) allocSoftmaxRegression(identifier string) {
 m.mem = C.mlpackGetSoftmaxRegressionPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *SoftmaxRegression) getSoftmaxRegression(identifier string) {
 m.allocSoftmaxRegression(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setSoftmaxRegression(identifier string, ptr *SoftmaxRegression) {
 C.mlpackSetSoftmaxRegressionPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program performs softmax regression, a generalization of logistic
  regression to the multiclass case, and has support for L2 regularization.  The
  program is able to train a model, load  an existing model, and give
  predictions (and optionally their accuracy) for test data.
  
  Training a softmax regression model is done by giving a file of training
  points with the 'training' parameter and their corresponding labels with the
  'labels' parameter. The number of classes can be manually specified with the
  'number_of_classes' parameter, and the maximum number of iterations of the
  L-BFGS optimizer can be specified with the 'max_iterations' parameter.  The L2
  regularization constant can be specified with the 'lambda' parameter and if an
  intercept term is not desired in the model, the 'no_intercept' parameter can
  be specified.
  
  The trained model can be saved with the 'output_model' output parameter. If
  training is not desired, but only testing is, a model can be loaded with the
  'input_model' parameter.  At the current time, a loaded model cannot be
  trained further, so specifying both 'input_model' and 'training' is not
  allowed.
  
  The program is also able to evaluate a model on test data.  A test dataset can
  be specified with the 'test' parameter. Class predictions can be saved with
  the 'predictions' output parameter.  If labels are specified for the test data
  with the 'test_labels' parameter, then the program will print the accuracy of
  the predictions on the given test set and its corresponding labels.
  
  For example, to train a softmax regression model on the data dataset with
  labels labels with a maximum of 1000 iterations for training, saving the
  trained model to sr_model, the following command can be used: 
  
  param := InitializeSoftmax_regression()
  param.Training = dataset
  param.Labels = labels
  sr_model, _ := Softmax_regression(param)
  
  Then, to use sr_model to classify the test points in test_points, saving the
  output predictions to predictions, the following command can be used:
  
  param := InitializeSoftmax_regression()
  param.Input_model = sr_model
  param.Test = test_points
  _, predictions := Softmax_regression(param)


  Input parameters:

   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - input_model (SoftmaxRegression): File containing existing model
        (parameters).
   - labels (mat.VecDense): A matrix containing labels (0 or 1) for the
        points in the training set (y). The labels must order as a row.
   - lambda (float64): L2-regularization constant  Default value 0.0001.
   - max_iterations (int): Maximum number of iterations before
        termination.  Default value 400.
   - no_intercept (bool): Do not add the intercept term to the model.
   - number_of_classes (int): Number of classes for classification; if
        unspecified (or 0), the number of classes found in the labels will be
        used.  Default value 0.
   - test (mat.Dense): Matrix containing test dataset.
   - test_labels (mat.VecDense): Matrix containing test labels.
   - training (mat.Dense): A matrix containing the training set (the
        matrix of predictors, X).
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output_model (SoftmaxRegression): File to save trained softmax
        regression model to.
   - predictions (mat.VecDense): Matrix to save predictions for test
        dataset into.

*/
func Softmax_regression(param *Softmax_regressionOptionalParam) (SoftmaxRegression, *mat.VecDense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Softmax Regression")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setSoftmaxRegression("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    GonumToArmaUrow("labels", param.Labels)
    SetPassed("labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda != 0.0001 {
    SetParamDouble("lambda", param.Lambda)
    SetPassed("lambda")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max_iterations != 400 {
    SetParamInt("max_iterations", param.Max_iterations)
    SetPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.No_intercept != false {
    SetParamBool("no_intercept", param.No_intercept)
    SetPassed("no_intercept")
  }

  // Detect if the parameter was passed; set if so.
  if param.Number_of_classes != 0 {
    SetParamInt("number_of_classes", param.Number_of_classes)
    SetPassed("number_of_classes")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    GonumToArmaMat("test", param.Test)
    SetPassed("test")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test_labels != nil {
    GonumToArmaUrow("test_labels", param.Test_labels)
    SetPassed("test_labels")
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
  SetPassed("output_model")
  SetPassed("predictions")

  // Call the mlpack program.
  C.mlpacksoftmax_regression()

  // Initialize result variable and get output.
  var output_model SoftmaxRegression
  output_model.getSoftmaxRegression("output_model")
  var predictions_ptr mlpackArma
  predictions := predictions_ptr.ArmaToGonumUrow("predictions")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output_model, predictions
}
