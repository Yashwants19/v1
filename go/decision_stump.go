package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_decision_stump
#include <capi/decision_stump.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type Decision_stumpOptionalParam struct {
    Bucket_size int
    Copy_all_inputs bool
    Input_model *DSModel
    Labels *mat.VecDense
    Test *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func InitializeDecision_stump() *Decision_stumpOptionalParam {
  return &Decision_stumpOptionalParam{
    Bucket_size: 6,
    Copy_all_inputs: false,
    Input_model: nil,
    Labels: nil,
    Test: nil,
    Training: nil,
    Verbose: false,
  }
}

type DSModel struct {
 mem unsafe.Pointer
}

func (m *DSModel) allocDSModel(identifier string) {
 m.mem = C.mlpackGetDSModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *DSModel) getDSModel(identifier string) {
 m.allocDSModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setDSModel(identifier string, ptr *DSModel) {
 C.mlpackSetDSModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program implements a decision stump, which is a single-level decision
  tree.  The decision stump will split on one dimension of the input data, and
  will split into multiple buckets.  The dimension and bins are selected by
  maximizing the information gain of the split.  Optionally, the minimum number
  of training points in each bin can be specified with the 'bucket_size'
  parameter.
  
  The decision stump is parameterized by a splitting dimension and a vector of
  values that denote the splitting values of each bin.
  
  This program enables several applications: a decision tree may be trained or
  loaded, and then that decision tree may be used to classify a given set of
  test points.  The decision tree may also be saved to a file for later usage.
  
  To train a decision stump, training data should be passed with the 'training'
  parameter, and their corresponding labels should be passed with the 'labels'
  option.  Optionally, if 'labels' is not specified, the labels are assumed to
  be the last dimension of the training dataset.  The 'bucket_size' parameter
  controls the minimum number of training points in each decision stump bucket.
  
  For classifying a test set, a decision stump may be loaded with the
  'input_model' parameter (useful for the situation where a stump has already
  been trained), and a test set may be specified with the 'test' parameter.  The
  predicted labels can be saved with the 'predictions' output parameter.
  
  Because decision stumps are trained in batch, retraining does not make sense
  and thus it is not possible to pass both 'training' and 'input_model';
  instead, simply build a new decision stump with the training data.
  
  After training, a decision stump can be saved with the 'output_model' output
  parameter.  That stump may later be re-used in subsequent calls to this
  program (or others).


  Input parameters:

   - bucket_size (int): The minimum number of training points in each
        decision stump bucket.  Default value 6.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - input_model (DSModel): Decision stump model to load.
   - labels (mat.VecDense): Labels for the training set. If not specified,
        the labels are assumed to be the last row of the training data.
   - test (mat.Dense): A dataset to calculate predictions for.
   - training (mat.Dense): The dataset to train on.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output_model (DSModel): Output decision stump model to save.
   - predictions (mat.VecDense): The output matrix that will hold the
        predicted labels for the test set.

*/
func Decision_stump(param *Decision_stumpOptionalParam) (DSModel, *mat.VecDense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Decision Stump")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Bucket_size != 6 {
    SetParamInt("bucket_size", param.Bucket_size)
    SetPassed("bucket_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setDSModel("input_model", param.Input_model)
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
  SetPassed("output_model")
  SetPassed("predictions")

  // Call the mlpack program.
  C.mlpackdecision_stump()

  // Initialize result variable and get output.
  var output_model DSModel
  output_model.getDSModel("output_model")
  var predictions_ptr mlpackArma
  predictions := predictions_ptr.ArmaToGonumUrow("predictions")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output_model, predictions
}
