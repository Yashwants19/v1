package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_adaboost
#include <capi/adaboost.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type AdaboostOptionalParam struct {
    Copy_all_inputs bool
    Input_model *AdaBoostModel
    Iterations int
    Labels *mat.VecDense
    Test *mat.Dense
    Tolerance float64
    Training *mat.Dense
    Verbose bool
    Weak_learner string
}

func InitializeAdaboost() *AdaboostOptionalParam {
  return &AdaboostOptionalParam{
    Copy_all_inputs: false,
    Input_model: nil,
    Iterations: 1000,
    Labels: nil,
    Test: nil,
    Tolerance: 1e-10,
    Training: nil,
    Verbose: false,
    Weak_learner: "decision_stump",
  }
}

type AdaBoostModel struct {
 mem unsafe.Pointer
}

func (m *AdaBoostModel) allocAdaBoostModel(identifier string) {
 m.mem = C.mlpackGetAdaBoostModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *AdaBoostModel) getAdaBoostModel(identifier string) {
 m.allocAdaBoostModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setAdaBoostModel(identifier string, ptr *AdaBoostModel) {
 C.mlpackSetAdaBoostModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program implements the AdaBoost (or Adaptive Boosting) algorithm. The
  variant of AdaBoost implemented here is AdaBoost.MH. It uses a weak learner,
  either decision stumps or perceptrons, and over many iterations, creates a
  strong learner that is a weighted ensemble of weak learners. It runs these
  iterations until a tolerance value is crossed for change in the value of the
  weighted training error.
  
  For more information about the algorithm, see the paper "Improved Boosting
  Algorithms Using Confidence-Rated Predictions", by R.E. Schapire and Y.
  Singer.
  
  This program allows training of an AdaBoost model, and then application of
  that model to a test dataset.  To train a model, a dataset must be passed with
  the 'training' option.  Labels can be given with the 'labels' option; if no
  labels are specified, the labels will be assumed to be the last column of the
  input dataset.  Alternately, an AdaBoost model may be loaded with the
  'input_model' option.
  
  Once a model is trained or loaded, it may be used to provide class predictions
  for a given test dataset.  A test dataset may be specified with the 'test'
  parameter.  The predicted classes for each point in the test dataset are
  output to the 'predictions' output parameter.  The AdaBoost model itself is
  output to the 'output_model' output parameter.
  
  Note: the following parameter is deprecated and will be removed in mlpack
  4.0.0: 'output'.
  Use 'predictions' instead of 'output'.
  
  For example, to run AdaBoost on an input dataset data with perceptrons as the
  weak learner type, storing the trained model in model, one could use the
  following command: 
  
  param := InitializeAdaboost()
  param.Training = data
  param.Weak_learner = "perceptron"
  _, model, _ := Adaboost(param)
  
  Similarly, an already-trained model in model can be used to provide class
  predictions from test data test_data and store the output in predictions with
  the following command: 
  
  param := InitializeAdaboost()
  param.Input_model = model
  param.Test = test_data
  _, _, predictions := Adaboost(param)


  Input parameters:

   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - input_model (AdaBoostModel): Input AdaBoost model.
   - iterations (int): The maximum number of boosting iterations to be run
        (0 will run until convergence.)  Default value 1000.
   - labels (mat.VecDense): Labels for the training set.
   - test (mat.Dense): Test dataset.
   - tolerance (float64): The tolerance for change in values of the
        weighted error during training.  Default value 1e-10.
   - training (mat.Dense): Dataset for training AdaBoost.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.
   - weak_learner (string): The type of weak learner to use:
        'decision_stump', or 'perceptron'.  Default value 'decision_stump'.

  Output parameters:

   - output (mat.VecDense): Predicted labels for the test set.
   - output_model (AdaBoostModel): Output trained AdaBoost model.
   - predictions (mat.VecDense): Predicted labels for the test set.

*/
func Adaboost(param *AdaboostOptionalParam) (*mat.VecDense, AdaBoostModel, *mat.VecDense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("AdaBoost")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setAdaBoostModel("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Iterations != 1000 {
    SetParamInt("iterations", param.Iterations)
    SetPassed("iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    GonumToArmaUrow("labels", param.Labels)
    SetPassed("labels")
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

  // Detect if the parameter was passed; set if so.
  if param.Weak_learner != "decision_stump" {
    SetParamString("weak_learner", param.Weak_learner)
    SetPassed("weak_learner")
  }

  // Mark all output options as passed.
  SetPassed("output")
  SetPassed("output_model")
  SetPassed("predictions")

  // Call the mlpack program.
  C.mlpackadaboost()

  // Initialize result variable and get output.
  var output_ptr mlpackArma
  output := output_ptr.ArmaToGonumUrow("output")
  var output_model AdaBoostModel
  output_model.getAdaBoostModel("output_model")
  var predictions_ptr mlpackArma
  predictions := predictions_ptr.ArmaToGonumUrow("predictions")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output, output_model, predictions
}
