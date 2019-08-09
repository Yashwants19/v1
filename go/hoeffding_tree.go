package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_hoeffding_tree
#include <capi/hoeffding_tree.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type Hoeffding_treeOptionalParam struct {
    Batch_mode bool
    Bins int
    Confidence float64
    Copy_all_inputs bool
    Info_gain bool
    Input_model *HoeffdingTreeModel
    Labels *mat.VecDense
    Max_samples int
    Min_samples int
    Numeric_split_strategy string
    Observations_before_binning int
    Passes int
    Test *DataWithInfo
    Test_labels *mat.VecDense
    Training *DataWithInfo
    Verbose bool
}

func InitializeHoeffding_tree() *Hoeffding_treeOptionalParam {
  return &Hoeffding_treeOptionalParam{
    Batch_mode: false,
    Bins: 10,
    Confidence: 0.95,
    Copy_all_inputs: false,
    Info_gain: false,
    Input_model: nil,
    Labels: nil,
    Max_samples: 5000,
    Min_samples: 100,
    Numeric_split_strategy: "binary",
    Observations_before_binning: 100,
    Passes: 1,
    Test: nil,
    Test_labels: nil,
    Training: nil,
    Verbose: false,
  }
}

type HoeffdingTreeModel struct {
 mem unsafe.Pointer
}

func (m *HoeffdingTreeModel) allocHoeffdingTreeModel(identifier string) {
 m.mem = C.mlpackGetHoeffdingTreeModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *HoeffdingTreeModel) getHoeffdingTreeModel(identifier string) {
 m.allocHoeffdingTreeModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setHoeffdingTreeModel(identifier string, ptr *HoeffdingTreeModel) {
 C.mlpackSetHoeffdingTreeModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program implements Hoeffding trees, a form of streaming decision tree
  suited best for large (or streaming) datasets.  This program supports both
  categorical and numeric data.  Given an input dataset, this program is able to
  train the tree with numerous training options, and save the model to a file. 
  The program is also able to use a trained model or a model from file in order
  to predict classes for a given test set.
  
  The training file and associated labels are specified with the 'training' and
  'labels' parameters, respectively. Optionally, if 'labels' is not specified,
  the labels are assumed to be the last dimension of the training dataset.
  
  The training may be performed in batch mode (like a typical decision tree
  algorithm) by specifying the 'batch_mode' option, but this may not be the best
  option for large datasets.
  
  When a model is trained, it may be saved via the 'output_model' output
  parameter.  A model may be loaded from file for further training or testing
  with the 'input_model' parameter.
  
  Test data may be specified with the 'test' parameter, and if performance
  statistics are desired for that test set, labels may be specified with the
  'test_labels' parameter.  Predictions for each test point may be saved with
  the 'predictions' output parameter, and class probabilities for each
  prediction may be saved with the 'probabilities' output parameter.
  
  For example, to train a Hoeffding tree with confidence 0.99 with data dataset,
  saving the trained tree to tree, the following command may be used:
  
  param := InitializeHoeffding_tree()
  param.Training = dataset
  param.Confidence = 0.99
  tree, _, _ := Hoeffding_tree(param)
  
  Then, this tree may be used to make predictions on the test set test_set,
  saving the predictions into predictions and the class probabilities into
  class_probs with the following command: 
  
  param := InitializeHoeffding_tree()
  param.Input_model = tree
  param.Test = test_set
  _, predictions, class_probs := Hoeffding_tree(param)


  Input parameters:

   - batch_mode (bool): If true, samples will be considered in batch
        instead of as a stream.  This generally results in better trees but at
        the cost of memory usage and runtime.
   - bins (int): If the 'domingos' split strategy is used, this specifies
        the number of bins for each numeric split.  Default value 10.
   - confidence (float64): Confidence before splitting (between 0 and 1). 
        Default value 0.95.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - info_gain (bool): If set, information gain is used instead of Gini
        impurity for calculating Hoeffding bounds.
   - input_model (HoeffdingTreeModel): Input trained Hoeffding tree
        model.
   - labels (mat.VecDense): Labels for training dataset.
   - max_samples (int): Maximum number of samples before splitting. 
        Default value 5000.
   - min_samples (int): Minimum number of samples before splitting. 
        Default value 100.
   - numeric_split_strategy (string): The splitting strategy to use for
        numeric features: 'domingos' or 'binary'.  Default value 'binary'.
   - observations_before_binning (int): If the 'domingos' split strategy
        is used, this specifies the number of samples observed before binning is
        performed.  Default value 100.
   - passes (int): Number of passes to take over the dataset.  Default
        value 1.
   - test (DataWithInfo): Testing dataset (may be categorical).
   - test_labels (mat.VecDense): Labels of test data.
   - training (DataWithInfo): Training dataset (may be categorical).
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output_model (HoeffdingTreeModel): Output for trained Hoeffding tree
        model.
   - predictions (mat.VecDense): Matrix to output label predictions for
        test data into.
   - probabilities (mat.Dense): In addition to predicting labels, provide
        rediction probabilities in this matrix.

*/
func Hoeffding_tree(param *Hoeffding_treeOptionalParam) (HoeffdingTreeModel, *mat.VecDense, *mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Hoeffding trees")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Batch_mode != false {
    SetParamBool("batch_mode", param.Batch_mode)
    SetPassed("batch_mode")
  }

  // Detect if the parameter was passed; set if so.
  if param.Bins != 10 {
    SetParamInt("bins", param.Bins)
    SetPassed("bins")
  }

  // Detect if the parameter was passed; set if so.
  if param.Confidence != 0.95 {
    SetParamDouble("confidence", param.Confidence)
    SetPassed("confidence")
  }

  // Detect if the parameter was passed; set if so.
  if param.Info_gain != false {
    SetParamBool("info_gain", param.Info_gain)
    SetPassed("info_gain")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setHoeffdingTreeModel("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    GonumToArmaUrow("labels", param.Labels)
    SetPassed("labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max_samples != 5000 {
    SetParamInt("max_samples", param.Max_samples)
    SetPassed("max_samples")
  }

  // Detect if the parameter was passed; set if so.
  if param.Min_samples != 100 {
    SetParamInt("min_samples", param.Min_samples)
    SetPassed("min_samples")
  }

  // Detect if the parameter was passed; set if so.
  if param.Numeric_split_strategy != "binary" {
    SetParamString("numeric_split_strategy", param.Numeric_split_strategy)
    SetPassed("numeric_split_strategy")
  }

  // Detect if the parameter was passed; set if so.
  if param.Observations_before_binning != 100 {
    SetParamInt("observations_before_binning", param.Observations_before_binning)
    SetPassed("observations_before_binning")
  }

  // Detect if the parameter was passed; set if so.
  if param.Passes != 1 {
    SetParamInt("passes", param.Passes)
    SetPassed("passes")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    GonumToArmaMatWithInfo("test", param.Test)
    SetPassed("test")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test_labels != nil {
    GonumToArmaUrow("test_labels", param.Test_labels)
    SetPassed("test_labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training != nil {
    GonumToArmaMatWithInfo("training", param.Training)
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
  SetPassed("probabilities")

  // Call the mlpack program.
  C.mlpackhoeffding_tree()

  // Initialize result variable and get output.
  var output_model HoeffdingTreeModel
  output_model.getHoeffdingTreeModel("output_model")
  var predictions_ptr mlpackArma
  predictions := predictions_ptr.ArmaToGonumUrow("predictions")
  var probabilities_ptr mlpackArma
  probabilities := probabilities_ptr.ArmaToGonumMat("probabilities")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output_model, predictions, probabilities
}
