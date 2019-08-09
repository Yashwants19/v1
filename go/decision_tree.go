package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_decision_tree
#include <capi/decision_tree.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type Decision_treeOptionalParam struct {
    Copy_all_inputs bool
    Input_model *DecisionTreeModel
    Labels *mat.VecDense
    Maximum_depth int
    Minimum_gain_split float64
    Minimum_leaf_size int
    Print_training_accuracy bool
    Print_training_error bool
    Test *DataWithInfo
    Test_labels *mat.VecDense
    Training *DataWithInfo
    Verbose bool
    Weights *mat.Dense
}

func InitializeDecision_tree() *Decision_treeOptionalParam {
  return &Decision_treeOptionalParam{
    Copy_all_inputs: false,
    Input_model: nil,
    Labels: nil,
    Maximum_depth: 0,
    Minimum_gain_split: 1e-07,
    Minimum_leaf_size: 20,
    Print_training_accuracy: false,
    Print_training_error: false,
    Test: nil,
    Test_labels: nil,
    Training: nil,
    Verbose: false,
    Weights: nil,
  }
}

type DecisionTreeModel struct {
 mem unsafe.Pointer
}

func (m *DecisionTreeModel) allocDecisionTreeModel(identifier string) {
 m.mem = C.mlpackGetDecisionTreeModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *DecisionTreeModel) getDecisionTreeModel(identifier string) {
 m.allocDecisionTreeModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setDecisionTreeModel(identifier string, ptr *DecisionTreeModel) {
 C.mlpackSetDecisionTreeModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  Train and evaluate using a decision tree.  Given a dataset containing numeric
  or categorical features, and associated labels for each point in the dataset,
  this program can train a decision tree on that data.
  
  The training set and associated labels are specified with the 'training' and
  'labels' parameters, respectively.  The labels should be in the range [0,
  num_classes - 1]. Optionally, if 'labels' is not specified, the labels are
  assumed to be the last dimension of the training dataset.
  
  When a model is trained, the 'output_model' output parameter may be used to
  save the trained model.  A model may be loaded for predictions with the
  'input_model' parameter.  The 'input_model' parameter may not be specified
  when the 'training' parameter is specified.  The 'minimum_leaf_size' parameter
  specifies the minimum number of training points that must fall into each leaf
  for it to be split.  The 'minimum_gain_split' parameter specifies the minimum
  gain that is needed for the node to split.  The 'maximum_depth' parameter
  specifies the maximum depth of the tree.  If 'print_training_error' is
  specified, the training error will be printed.
  
  Test data may be specified with the 'test' parameter, and if performance
  numbers are desired for that test set, labels may be specified with the
  'test_labels' parameter.  Predictions for each test point may be saved via the
  'predictions' output parameter.  Class probabilities for each prediction may
  be saved with the 'probabilities' output parameter.
  
  For example, to train a decision tree with a minimum leaf size of 20 on the
  dataset contained in data with labels labels, saving the output model to tree
  and printing the training error, one could call
  
  param := InitializeDecision_tree()
  param.Training = data
  param.Labels = labels
  param.Minimum_leaf_size = 20
  param.Minimum_gain_split = 0.001
  param.Print_training_accuracy = true
  tree, _, _ := Decision_tree(param)
  
  Then, to use that model to classify points in test_set and print the test
  error given the labels test_labels using that model, while saving the
  predictions for each point to predictions, one could call 
  
  param := InitializeDecision_tree()
  param.Input_model = tree
  param.Test = test_set
  param.Test_labels = test_labels
  _, predictions, _ := Decision_tree(param)


  Input parameters:

   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - input_model (DecisionTreeModel): Pre-trained decision tree, to be
        used with test points.
   - labels (mat.VecDense): Training labels.
   - maximum_depth (int): Maximum depth of the tree (0 means no limit). 
        Default value 0.
   - minimum_gain_split (float64): Minimum gain for node splitting. 
        Default value 1e-07.
   - minimum_leaf_size (int): Minimum number of points in a leaf.  Default
        value 20.
   - print_training_accuracy (bool): Print the training accuracy.
   - print_training_error (bool): Print the training error (deprecated;
        will be removed in mlpack 4.0.0).
   - test (DataWithInfo): Testing dataset (may be categorical).
   - test_labels (mat.VecDense): Test point labels, if accuracy
        calculation is desired.
   - training (DataWithInfo): Training dataset (may be categorical).
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.
   - weights (mat.Dense): The weight of labels

  Output parameters:

   - output_model (DecisionTreeModel): Output for trained decision tree.
   - predictions (mat.VecDense): Class predictions for each test point.
   - probabilities (mat.Dense): Class probabilities for each test point.

*/
func Decision_tree(param *Decision_treeOptionalParam) (DecisionTreeModel, *mat.VecDense, *mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Decision tree")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setDecisionTreeModel("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    GonumToArmaUrow("labels", param.Labels)
    SetPassed("labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Maximum_depth != 0 {
    SetParamInt("maximum_depth", param.Maximum_depth)
    SetPassed("maximum_depth")
  }

  // Detect if the parameter was passed; set if so.
  if param.Minimum_gain_split != 1e-07 {
    SetParamDouble("minimum_gain_split", param.Minimum_gain_split)
    SetPassed("minimum_gain_split")
  }

  // Detect if the parameter was passed; set if so.
  if param.Minimum_leaf_size != 20 {
    SetParamInt("minimum_leaf_size", param.Minimum_leaf_size)
    SetPassed("minimum_leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Print_training_accuracy != false {
    SetParamBool("print_training_accuracy", param.Print_training_accuracy)
    SetPassed("print_training_accuracy")
  }

  // Detect if the parameter was passed; set if so.
  if param.Print_training_error != false {
    SetParamBool("print_training_error", param.Print_training_error)
    SetPassed("print_training_error")
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

  // Detect if the parameter was passed; set if so.
  if param.Weights != nil {
    GonumToArmaMat("weights", param.Weights)
    SetPassed("weights")
  }

  // Mark all output options as passed.
  SetPassed("output_model")
  SetPassed("predictions")
  SetPassed("probabilities")

  // Call the mlpack program.
  C.mlpackdecision_tree()

  // Initialize result variable and get output.
  var output_model DecisionTreeModel
  output_model.getDecisionTreeModel("output_model")
  var predictions_ptr mlpackArma
  predictions := predictions_ptr.ArmaToGonumUrow("predictions")
  var probabilities_ptr mlpackArma
  probabilities := probabilities_ptr.ArmaToGonumMat("probabilities")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output_model, predictions, probabilities
}
