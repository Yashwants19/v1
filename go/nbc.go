package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_nbc
#include <capi/nbc.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type NbcOptionalParam struct {
    Copy_all_inputs bool
    Incremental_variance bool
    Input_model *NBCModel
    Labels *mat.VecDense
    Test *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func InitializeNbc() *NbcOptionalParam {
  return &NbcOptionalParam{
    Copy_all_inputs: false,
    Incremental_variance: false,
    Input_model: nil,
    Labels: nil,
    Test: nil,
    Training: nil,
    Verbose: false,
  }
}

type NBCModel struct {
 mem unsafe.Pointer
}

func (m *NBCModel) allocNBCModel(identifier string) {
 m.mem = C.mlpackGetNBCModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *NBCModel) getNBCModel(identifier string) {
 m.allocNBCModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setNBCModel(identifier string, ptr *NBCModel) {
 C.mlpackSetNBCModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program trains the Naive Bayes classifier on the given labeled training
  set, or loads a model from the given model file, and then may use that trained
  model to classify the points in a given test set.
  
  The training set is specified with the 'training' parameter.  Labels may be
  either the last row of the training set, or alternately the 'labels' parameter
  may be specified to pass a separate matrix of labels.
  
  If training is not desired, a pre-existing model may be loaded with the
  'input_model' parameter.
  
  
  
  The 'incremental_variance' parameter can be used to force the training to use
  an incremental algorithm for calculating variance.  This is slower, but can
  help avoid loss of precision in some cases.
  
  If classifying a test set is desired, the test set may be specified with the
  'test' parameter, and the classifications may be saved with the
  'predictions'predictions  parameter.  If saving the trained model is desired,
  this may be done with the 'output_model' output parameter.
  
  Note: the 'output' and 'output_probs' parameters are deprecated and will be
  removed in mlpack 4.0.0.  Use 'predictions' and 'probabilities' instead.
  
  For example, to train a Naive Bayes classifier on the dataset data with labels
  labels and save the model to nbc_model, the following command may be used:
  
  param := InitializeNbc()
  param.Training = data
  param.Labels = labels
  _, nbc_model, _, _, _ := Nbc(param)
  
  Then, to use nbc_model to predict the classes of the dataset test_set and save
  the predicted classes to predictions, the following command may be used:
  
  param := InitializeNbc()
  param.Input_model = nbc_model
  param.Test = test_set
  predictions, _, _, _, _ := Nbc(param)


  Input parameters:

   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - incremental_variance (bool): The variance of each class will be
        calculated incrementally.
   - input_model (NBCModel): Input Naive Bayes model.
   - labels (mat.VecDense): A file containing labels for the training
        set.
   - test (mat.Dense): A matrix containing the test set.
   - training (mat.Dense): A matrix containing the training set.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.VecDense): The matrix in which the predicted labels for
        the test set will be written (deprecated).
   - output_model (NBCModel): File to save trained Naive Bayes model to.
   - output_probs (mat.Dense): The matrix in which the predicted
        probability of labels for the test set will be written (deprecated).
   - predictions (mat.VecDense): The matrix in which the predicted labels
        for the test set will be written.
   - probabilities (mat.Dense): The matrix in which the predicted
        probability of labels for the test set will be written.

*/
func Nbc(param *NbcOptionalParam) (*mat.VecDense, NBCModel, *mat.Dense, *mat.VecDense, *mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Parametric Naive Bayes Classifier")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Incremental_variance != false {
    SetParamBool("incremental_variance", param.Incremental_variance)
    SetPassed("incremental_variance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setNBCModel("input_model", param.Input_model)
    SetPassed("input_model")
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
  SetPassed("output_probs")
  SetPassed("predictions")
  SetPassed("probabilities")

  // Call the mlpack program.
  C.mlpacknbc()

  // Initialize result variable and get output.
  var output_ptr mlpackArma
  output := output_ptr.ArmaToGonumUrow("output")
  var output_model NBCModel
  output_model.getNBCModel("output_model")
  var output_probs_ptr mlpackArma
  output_probs := output_probs_ptr.ArmaToGonumMat("output_probs")
  var predictions_ptr mlpackArma
  predictions := predictions_ptr.ArmaToGonumUrow("predictions")
  var probabilities_ptr mlpackArma
  probabilities := probabilities_ptr.ArmaToGonumMat("probabilities")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output, output_model, output_probs, predictions, probabilities
}
