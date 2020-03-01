package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_nbc
#include <capi/nbc.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "unsafe" 
)

type NbcOptionalParam struct {
    IncrementalVariance bool
    InputModel *nbcModel
    Labels *mat.Dense
    Test *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func InitializeNbc() *NbcOptionalParam {
  return &NbcOptionalParam{
    IncrementalVariance: false,
    InputModel: nil,
    Labels: nil,
    Test: nil,
    Training: nil,
    Verbose: false,
  }
}

type nbcModel struct {
  mem unsafe.Pointer
}

func (m *nbcModel) allocNBCModel(identifier string) {
  m.mem = C.mlpackGetNBCModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *nbcModel) getNBCModel(identifier string) {
  m.allocNBCModel(identifier)
}

func setNBCModel(identifier string, ptr *nbcModel) {
  C.mlpackSetNBCModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program trains the Naive Bayes classifier on the given labeled training
  set, or loads a model from the given model file, and then may use that trained
  model to classify the points in a given test set.
  
  The training set is specified with the "training" parameter.  Labels may be
  either the last row of the training set, or alternately the "labels" parameter
  may be specified to pass a separate matrix of labels.
  
  If training is not desired, a pre-existing model may be loaded with the
  "input_model" parameter.
  
  
  
  The "incremental_variance" parameter can be used to force the training to use
  an incremental algorithm for calculating variance.  This is slower, but can
  help avoid loss of precision in some cases.
  
  If classifying a test set is desired, the test set may be specified with the
  "test" parameter, and the classifications may be saved with the
  "predictions"predictions  parameter.  If saving the trained model is desired,
  this may be done with the "output_model" output parameter.
  
  Note: the "output" and "output_probs" parameters are deprecated and will be
  removed in mlpack 4.0.0.  Use "predictions" and "probabilities" instead.
  
  For example, to train a Naive Bayes classifier on the dataset data with labels
  labels and save the model to nbc_model, the following command may be used:
  
    param := mlpack.InitializeNbc()
    param.Training = data
    param.Labels = labels
    _, NbcModel, _, _, _ := mlpack.Nbc(param)
  
  Then, to use nbc_model to predict the classes of the dataset test_set and save
  the predicted classes to predictions, the following command may be used:
  
    param := mlpack.InitializeNbc()
    param.InputModel = &NbcModel
    param.Test = test_set
    Predictions, _, _, _, _ := mlpack.Nbc(param)


  Input parameters:

   - IncrementalVariance (bool): The variance of each class will be
        calculated incrementally.
   - InputModel (nbcModel): Input Naive Bayes model.
   - Labels (mat.Dense): A file containing labels for the training set.
   - Test (mat.Dense): A matrix containing the test set.
   - Training (mat.Dense): A matrix containing the training set.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - Output (mat.Dense): The matrix in which the predicted labels for the
        test set will be written (deprecated).
   - OutputModel (nbcModel): File to save trained Naive Bayes model to.
   - OutputProbs (mat.Dense): The matrix in which the predicted
        probability of labels for the test set will be written (deprecated).
   - Predictions (mat.Dense): The matrix in which the predicted labels for
        the test set will be written.
   - Probabilities (mat.Dense): The matrix in which the predicted
        probability of labels for the test set will be written.

 */
func Nbc(param *NbcOptionalParam) (*mat.Dense, nbcModel, *mat.Dense, *mat.Dense, *mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Parametric Naive Bayes Classifier")

  // Detect if the parameter was passed; set if so.
  if param.IncrementalVariance != false {
    setParamBool("incremental_variance", param.IncrementalVariance)
    setPassed("incremental_variance")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setNBCModel("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    gonumToArmaUrow("labels", param.Labels)
    setPassed("labels")
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
  setPassed("output_probs")
  setPassed("predictions")
  setPassed("probabilities")

  // Call the mlpack program.
  C.mlpackNbc()

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  Output := outputPtr.armaToGonumUrow("output")
  var OutputModel nbcModel
  OutputModel.getNBCModel("output_model")
  var outputProbsPtr mlpackArma
  OutputProbs := outputProbsPtr.armaToGonumMat("output_probs")
  var predictionsPtr mlpackArma
  Predictions := predictionsPtr.armaToGonumUrow("predictions")
  var probabilitiesPtr mlpackArma
  Probabilities := probabilitiesPtr.armaToGonumMat("probabilities")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return Output, OutputModel, OutputProbs, Predictions, Probabilities
}
