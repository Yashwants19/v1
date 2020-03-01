package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_softmax_regression
#include <capi/softmax_regression.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "unsafe" 
)

type SoftmaxRegressionOptionalParam struct {
    InputModel *softmaxRegression
    Labels *mat.Dense
    Lambda float64
    MaxIterations int
    NoIntercept bool
    NumberOfClasses int
    Test *mat.Dense
    TestLabels *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func InitializeSoftmaxRegression() *SoftmaxRegressionOptionalParam {
  return &SoftmaxRegressionOptionalParam{
    InputModel: nil,
    Labels: nil,
    Lambda: 0.0001,
    MaxIterations: 400,
    NoIntercept: false,
    NumberOfClasses: 0,
    Test: nil,
    TestLabels: nil,
    Training: nil,
    Verbose: false,
  }
}

type softmaxRegression struct {
  mem unsafe.Pointer
}

func (m *softmaxRegression) allocSoftmaxRegression(identifier string) {
  m.mem = C.mlpackGetSoftmaxRegressionPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *softmaxRegression) getSoftmaxRegression(identifier string) {
  m.allocSoftmaxRegression(identifier)
}

func setSoftmaxRegression(identifier string, ptr *softmaxRegression) {
  C.mlpackSetSoftmaxRegressionPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program performs softmax regression, a generalization of logistic
  regression to the multiclass case, and has support for L2 regularization.  The
  program is able to train a model, load  an existing model, and give
  predictions (and optionally their accuracy) for test data.
  
  Training a softmax regression model is done by giving a file of training
  points with the "training" parameter and their corresponding labels with the
  "labels" parameter. The number of classes can be manually specified with the
  "number_of_classes" parameter, and the maximum number of iterations of the
  L-BFGS optimizer can be specified with the "max_iterations" parameter.  The L2
  regularization constant can be specified with the "lambda" parameter and if an
  intercept term is not desired in the model, the "no_intercept" parameter can
  be specified.
  
  The trained model can be saved with the "output_model" output parameter. If
  training is not desired, but only testing is, a model can be loaded with the
  "input_model" parameter.  At the current time, a loaded model cannot be
  trained further, so specifying both "input_model" and "training" is not
  allowed.
  
  The program is also able to evaluate a model on test data.  A test dataset can
  be specified with the "test" parameter. Class predictions can be saved with
  the "predictions" output parameter.  If labels are specified for the test data
  with the "test_labels" parameter, then the program will print the accuracy of
  the predictions on the given test set and its corresponding labels.
  
  For example, to train a softmax regression model on the data dataset with
  labels labels with a maximum of 1000 iterations for training, saving the
  trained model to sr_model, the following command can be used: 
  
    param := mlpack.InitializeSoftmaxRegression()
    param.Training = dataset
    param.Labels = labels
    SrModel, _ := mlpack.SoftmaxRegression(param)
  
  Then, to use sr_model to classify the test points in test_points, saving the
  output predictions to predictions, the following command can be used:
  
    param := mlpack.InitializeSoftmaxRegression()
    param.InputModel = &SrModel
    param.Test = test_points
    _, Predictions := mlpack.SoftmaxRegression(param)


  Input parameters:

   - InputModel (softmaxRegression): File containing existing model
        (parameters).
   - Labels (mat.Dense): A matrix containing labels (0 or 1) for the
        points in the training set (y). The labels must order as a row.
   - Lambda (float64): L2-regularization constant  Default value 0.0001.
   - MaxIterations (int): Maximum number of iterations before termination.
         Default value 400.
   - NoIntercept (bool): Do not add the intercept term to the model.
   - NumberOfClasses (int): Number of classes for classification; if
        unspecified (or 0), the number of classes found in the labels will be
        used.  Default value 0.
   - Test (mat.Dense): Matrix containing test dataset.
   - TestLabels (mat.Dense): Matrix containing test labels.
   - Training (mat.Dense): A matrix containing the training set (the
        matrix of predictors, X).
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - OutputModel (softmaxRegression): File to save trained softmax
        regression model to.
   - Predictions (mat.Dense): Matrix to save predictions for test dataset
        into.

 */
func SoftmaxRegression(param *SoftmaxRegressionOptionalParam) (softmaxRegression, *mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Softmax Regression")

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setSoftmaxRegression("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    gonumToArmaUrow("labels", param.Labels)
    setPassed("labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda != 0.0001 {
    setParamDouble("lambda", param.Lambda)
    setPassed("lambda")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 400 {
    setParamInt("max_iterations", param.MaxIterations)
    setPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.NoIntercept != false {
    setParamBool("no_intercept", param.NoIntercept)
    setPassed("no_intercept")
  }

  // Detect if the parameter was passed; set if so.
  if param.NumberOfClasses != 0 {
    setParamInt("number_of_classes", param.NumberOfClasses)
    setPassed("number_of_classes")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMat("test", param.Test)
    setPassed("test")
  }

  // Detect if the parameter was passed; set if so.
  if param.TestLabels != nil {
    gonumToArmaUrow("test_labels", param.TestLabels)
    setPassed("test_labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training != nil {
    gonumToArmaMat("training", param.Training)
    setPassed("training")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("output_model")
  setPassed("predictions")

  // Call the mlpack program.
  C.mlpackSoftmaxRegression()

  // Initialize result variable and get output.
  var OutputModel softmaxRegression
  OutputModel.getSoftmaxRegression("output_model")
  var predictionsPtr mlpackArma
  Predictions := predictionsPtr.armaToGonumUrow("predictions")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return OutputModel, Predictions
}
