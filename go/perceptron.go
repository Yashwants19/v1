package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_perceptron
#include <capi/perceptron.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type PerceptronOptionalParam struct {
    Copy_all_inputs bool
    Input_model *PerceptronModel
    Labels *mat.VecDense
    Max_iterations int
    Test *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func InitializePerceptron() *PerceptronOptionalParam {
  return &PerceptronOptionalParam{
    Copy_all_inputs: false,
    Input_model: nil,
    Labels: nil,
    Max_iterations: 1000,
    Test: nil,
    Training: nil,
    Verbose: false,
  }
}

type PerceptronModel struct {
 mem unsafe.Pointer
}

func (m *PerceptronModel) allocPerceptronModel(identifier string) {
 m.mem = C.mlpackGetPerceptronModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *PerceptronModel) getPerceptronModel(identifier string) {
 m.allocPerceptronModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setPerceptronModel(identifier string, ptr *PerceptronModel) {
 C.mlpackSetPerceptronModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program implements a perceptron, which is a single level neural network.
  The perceptron makes its predictions based on a linear predictor function
  combining a set of weights with the feature vector.  The perceptron learning
  rule is able to converge, given enough iterations (specified using the
  'max_iterations' parameter), if the data supplied is linearly separable.  The
  perceptron is parameterized by a matrix of weight vectors that denote the
  numerical weights of the neural network.
  
  This program allows loading a perceptron from a model (via the 'input_model'
  parameter) or training a perceptron given training data (via the 'training'
  parameter), or both those things at once.  In addition, this program allows
  classification on a test dataset (via the 'test' parameter) and the
  classification results on the test set may be saved with the 'predictions'
  output parameter.  The perceptron model may be saved with the 'output_model'
  output parameter.
  
  Note: the following parameter is deprecated and will be removed in mlpack
  4.0.0: 'output'.
  Use 'predictions' instead of 'output'.
  
  The training data given with the 'training' option may have class labels as
  its last dimension (so, if the training data is in CSV format, labels should
  be the last column).  Alternately, the 'labels' parameter may be used to
  specify a separate matrix of labels.
  
  All these options make it easy to train a perceptron, and then re-use that
  perceptron for later classification.  The invocation below trains a perceptron
  on training_data with labels training_labels, and saves the model to
  perceptron_model.
  
  param := InitializePerceptron()
  param.Training = training_data
  param.Labels = training_labels
  _, perceptron_model, _ := Perceptron(param)
  
  Then, this model can be re-used for classification on the test data test_data.
   The example below does precisely that, saving the predicted classes to
  predictions.
  
  param := InitializePerceptron()
  param.Input_model = perceptron_model
  param.Test = test_data
  _, _, predictions := Perceptron(param)
  
  Note that all of the options may be specified at once: predictions may be
  calculated right after training a model, and model training can occur even if
  an existing perceptron model is passed with the 'input_model' parameter. 
  However, note that the number of classes and the dimensionality of all data
  must match.  So you cannot pass a perceptron model trained on 2 classes and
  then re-train with a 4-class dataset.  Similarly, attempting classification on
  a 3-dimensional dataset with a perceptron that has been trained on 8
  dimensions will cause an error.


  Input parameters:

   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - input_model (PerceptronModel): Input perceptron model.
   - labels (mat.VecDense): A matrix containing labels for the training
        set.
   - max_iterations (int): The maximum number of iterations the perceptron
        is to be run  Default value 1000.
   - test (mat.Dense): A matrix containing the test set.
   - training (mat.Dense): A matrix containing the training set.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.VecDense): The matrix in which the predicted labels for
        the test set will be written.
   - output_model (PerceptronModel): Output for trained perceptron model.
   - predictions (mat.VecDense): The matrix in which the predicted labels
        for the test set will be written.

*/
func Perceptron(param *PerceptronOptionalParam) (*mat.VecDense, PerceptronModel, *mat.VecDense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Perceptron")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setPerceptronModel("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    GonumToArmaUrow("labels", param.Labels)
    SetPassed("labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max_iterations != 1000 {
    SetParamInt("max_iterations", param.Max_iterations)
    SetPassed("max_iterations")
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
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("output")
  SetPassed("output_model")
  SetPassed("predictions")

  // Call the mlpack program.
  C.mlpackperceptron()

  // Initialize result variable and get output.
  var output_ptr mlpackArma
  output := output_ptr.ArmaToGonumUrow("output")
  var output_model PerceptronModel
  output_model.getPerceptronModel("output_model")
  var predictions_ptr mlpackArma
  predictions := predictions_ptr.ArmaToGonumUrow("predictions")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output, output_model, predictions
}
