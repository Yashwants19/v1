package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_random_forest
#include <capi/random_forest.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type Random_forestOptionalParam struct {
    Copy_all_inputs bool
    Input_model *RandomForestModel
    Labels *mat.VecDense
    Maximum_depth int
    Minimum_gain_split float64
    Minimum_leaf_size int
    Num_trees int
    Print_training_accuracy bool
    Seed int
    Subspace_dim int
    Test *mat.Dense
    Test_labels *mat.VecDense
    Training *mat.Dense
    Verbose bool
}

func InitializeRandom_forest() *Random_forestOptionalParam {
  return &Random_forestOptionalParam{
    Copy_all_inputs: false,
    Input_model: nil,
    Labels: nil,
    Maximum_depth: 0,
    Minimum_gain_split: 0,
    Minimum_leaf_size: 1,
    Num_trees: 10,
    Print_training_accuracy: false,
    Seed: 0,
    Subspace_dim: 0,
    Test: nil,
    Test_labels: nil,
    Training: nil,
    Verbose: false,
  }
}

type RandomForestModel struct {
 mem unsafe.Pointer
}

func (m *RandomForestModel) allocRandomForestModel(identifier string) {
 m.mem = C.mlpackGetRandomForestModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *RandomForestModel) getRandomForestModel(identifier string) {
 m.allocRandomForestModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setRandomForestModel(identifier string, ptr *RandomForestModel) {
 C.mlpackSetRandomForestModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program is an implementation of the standard random forest classification
  algorithm by Leo Breiman.  A random forest can be trained and saved for later
  use, or a random forest may be loaded and predictions or class probabilities
  for points may be generated.
  
  The training set and associated labels are specified with the 'training' and
  'labels' parameters, respectively.  The labels should be in the range [0,
  num_classes - 1]. Optionally, if 'labels' is not specified, the labels are
  assumed to be the last dimension of the training dataset.
  
  When a model is trained, the 'output_model' output parameter may be used to
  save the trained model.  A model may be loaded for predictions with the
  'input_model'parameter. The 'input_model' parameter may not be specified when
  the 'training' parameter is specified.  The 'minimum_leaf_size' parameter
  specifies the minimum number of training points that must fall into each leaf
  for it to be split.  The 'num_trees' controls the number of trees in the
  random forest.  The 'minimum_gain_split' parameter controls the minimum
  required gain for a decision tree node to split.  Larger values will force
  higher-confidence splits.  The 'maximum_depth' parameter specifies the maximum
  depth of the tree.  The 'subspace_dim' parameter is used to control the number
  of random dimensions chosen for an individual node's split.  If
  'print_training_accuracy' is specified, the calculated accuracy on the
  training set will be printed.
  
  Test data may be specified with the 'test' parameter, and if performance
  measures are desired for that test set, labels for the test points may be
  specified with the 'test_labels' parameter.  Predictions for each test point
  may be saved via the 'predictions'output parameter.  Class probabilities for
  each prediction may be saved with the 'probabilities' output parameter.
  
  For example, to train a random forest with a minimum leaf size of 20 using 10
  trees on the dataset contained in datawith labels labels, saving the output
  random forest to rf_model and printing the training error, one could call
  
  param := InitializeRandom_forest()
  param.Training = data
  param.Labels = labels
  param.Minimum_leaf_size = 20
  param.Num_trees = 10
  param.Print_training_accuracy = true
  rf_model, _, _ := Random_forest(param)
  
  Then, to use that model to classify points in test_set and print the test
  error given the labels test_labels using that model, while saving the
  predictions for each point to predictions, one could call 
  
  param := InitializeRandom_forest()
  param.Input_model = rf_model
  param.Test = test_set
  param.Test_labels = test_labels
  _, predictions, _ := Random_forest(param)


  Input parameters:

   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - input_model (RandomForestModel): Pre-trained random forest to use for
        classification.
   - labels (mat.VecDense): Labels for training dataset.
   - maximum_depth (int): Maximum depth of the tree (0 means no limit). 
        Default value 0.
   - minimum_gain_split (float64): Minimum gain needed to make a split
        when building a tree.  Default value 0.
   - minimum_leaf_size (int): Minimum number of points in each leaf node. 
        Default value 1.
   - num_trees (int): Number of trees in the random forest.  Default value
        10.
   - print_training_accuracy (bool): If set, then the accuracy of the
        model on the training set will be predicted (verbose must also be
        specified).
   - seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - subspace_dim (int): Dimensionality of random subspace to use for each
        split.  '0' will autoselect the square root of data dimensionality. 
        Default value 0.
   - test (mat.Dense): Test dataset to produce predictions for.
   - test_labels (mat.VecDense): Test dataset labels, if accuracy
        calculation is desired.
   - training (mat.Dense): Training dataset.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output_model (RandomForestModel): Model to save trained random forest
        to.
   - predictions (mat.VecDense): Predicted classes for each point in the
        test set.
   - probabilities (mat.Dense): Predicted class probabilities for each
        point in the test set.

*/
func Random_forest(param *Random_forestOptionalParam) (RandomForestModel, *mat.VecDense, *mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Random forests")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setRandomForestModel("input_model", param.Input_model)
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
  if param.Minimum_gain_split != 0 {
    SetParamDouble("minimum_gain_split", param.Minimum_gain_split)
    SetPassed("minimum_gain_split")
  }

  // Detect if the parameter was passed; set if so.
  if param.Minimum_leaf_size != 1 {
    SetParamInt("minimum_leaf_size", param.Minimum_leaf_size)
    SetPassed("minimum_leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Num_trees != 10 {
    SetParamInt("num_trees", param.Num_trees)
    SetPassed("num_trees")
  }

  // Detect if the parameter was passed; set if so.
  if param.Print_training_accuracy != false {
    SetParamBool("print_training_accuracy", param.Print_training_accuracy)
    SetPassed("print_training_accuracy")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    SetParamInt("seed", param.Seed)
    SetPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Subspace_dim != 0 {
    SetParamInt("subspace_dim", param.Subspace_dim)
    SetPassed("subspace_dim")
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
  SetPassed("probabilities")

  // Call the mlpack program.
  C.mlpackrandom_forest()

  // Initialize result variable and get output.
  var output_model RandomForestModel
  output_model.getRandomForestModel("output_model")
  var predictions_ptr mlpackArma
  predictions := predictions_ptr.ArmaToGonumUrow("predictions")
  var probabilities_ptr mlpackArma
  probabilities := probabilities_ptr.ArmaToGonumMat("probabilities")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output_model, predictions, probabilities
}
