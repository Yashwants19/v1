package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_perceptron
#include <capi/perceptron.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "unsafe" 
)

type PerceptronOptionalParam struct {
    InputModel *perceptronModel
    Labels *mat.Dense
    MaxIterations int
    Test *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func InitializePerceptron() *PerceptronOptionalParam {
  return &PerceptronOptionalParam{
    InputModel: nil,
    Labels: nil,
    MaxIterations: 1000,
    Test: nil,
    Training: nil,
    Verbose: false,
  }
}

type perceptronModel struct {
  mem unsafe.Pointer
}

func (m *perceptronModel) allocPerceptronModel(identifier string) {
  m.mem = C.mlpackGetPerceptronModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *perceptronModel) getPerceptronModel(identifier string) {
  m.allocPerceptronModel(identifier)
}

func setPerceptronModel(identifier string, ptr *perceptronModel) {
  C.mlpackSetPerceptronModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program implements a perceptron, which is a single level neural network.
  The perceptron makes its predictions based on a linear predictor function
  combining a set of weights with the feature vector.  The perceptron learning
  rule is able to converge, given enough iterations (specified using the
  "max_iterations" parameter), if the data supplied is linearly separable.  The
  perceptron is parameterized by a matrix of weight vectors that denote the
  numerical weights of the neural network.
  
  This program allows loading a perceptron from a model (via the "input_model"
  parameter) or training a perceptron given training data (via the "training"
  parameter), or both those things at once.  In addition, this program allows
  classification on a test dataset (via the "test" parameter) and the
  classification results on the test set may be saved with the "predictions"
  output parameter.  The perceptron model may be saved with the "output_model"
  output parameter.
  
  Note: the following parameter is deprecated and will be removed in mlpack
  4.0.0: "output".
  Use "predictions" instead of "output".
  
  The training data given with the "training" option may have class labels as
  its last dimension (so, if the training data is in CSV format, labels should
  be the last column).  Alternately, the "labels" parameter may be used to
  specify a separate matrix of labels.
  
  All these options make it easy to train a perceptron, and then re-use that
  perceptron for later classification.  The invocation below trains a perceptron
  on training_data with labels training_labels, and saves the model to
  perceptron_model.
  
    param := mlpack.InitializePerceptron()
    param.Training = training_data
    param.Labels = training_labels
    _, PerceptronModel, _ := mlpack.Perceptron(param)
  
  Then, this model can be re-used for classification on the test data test_data.
   The example below does precisely that, saving the predicted classes to
  predictions.
  
    param := mlpack.InitializePerceptron()
    param.InputModel = &PerceptronModel
    param.Test = test_data
    _, _, Predictions := mlpack.Perceptron(param)
  
  Note that all of the options may be specified at once: predictions may be
  calculated right after training a model, and model training can occur even if
  an existing perceptron model is passed with the "input_model" parameter. 
  However, note that the number of classes and the dimensionality of all data
  must match.  So you cannot pass a perceptron model trained on 2 classes and
  then re-train with a 4-class dataset.  Similarly, attempting classification on
  a 3-dimensional dataset with a perceptron that has been trained on 8
  dimensions will cause an error.


  Input parameters:

   - InputModel (perceptronModel): Input perceptron model.
   - Labels (mat.Dense): A matrix containing labels for the training set.
   - MaxIterations (int): The maximum number of iterations the perceptron
        is to be run  Default value 1000.
   - Test (mat.Dense): A matrix containing the test set.
   - Training (mat.Dense): A matrix containing the training set.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - Output (mat.Dense): The matrix in which the predicted labels for the
        test set will be written.
   - OutputModel (perceptronModel): Output for trained perceptron model.
   - Predictions (mat.Dense): The matrix in which the predicted labels for
        the test set will be written.

 */
func Perceptron(param *PerceptronOptionalParam) (*mat.Dense, perceptronModel, *mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Perceptron")

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setPerceptronModel("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    gonumToArmaUrow("labels", param.Labels)
    setPassed("labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 1000 {
    setParamInt("max_iterations", param.MaxIterations)
    setPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMat("test", param.Test)
    setPassed("test")
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
  setPassed("output")
  setPassed("output_model")
  setPassed("predictions")

  // Call the mlpack program.
  C.mlpackPerceptron()

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  Output := outputPtr.armaToGonumUrow("output")
  var OutputModel perceptronModel
  OutputModel.getPerceptronModel("output_model")
  var predictionsPtr mlpackArma
  Predictions := predictionsPtr.armaToGonumUrow("predictions")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return Output, OutputModel, Predictions
}
