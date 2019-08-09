package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_det
#include <capi/det.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type DetOptionalParam struct {
    Copy_all_inputs bool
    Folds int
    Input_model *DTree
    Max_leaf_size int
    Min_leaf_size int
    Path_format string
    Skip_pruning bool
    Test *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func InitializeDet() *DetOptionalParam {
  return &DetOptionalParam{
    Copy_all_inputs: false,
    Folds: 10,
    Input_model: nil,
    Max_leaf_size: 10,
    Min_leaf_size: 5,
    Path_format: "lr",
    Skip_pruning: false,
    Test: nil,
    Training: nil,
    Verbose: false,
  }
}

type DTree struct {
 mem unsafe.Pointer
}

func (m *DTree) allocDTree(identifier string) {
 m.mem = C.mlpackGetDTreePtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *DTree) getDTree(identifier string) {
 m.allocDTree(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setDTree(identifier string, ptr *DTree) {
 C.mlpackSetDTreePtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program performs a number of functions related to Density Estimation
  Trees.  The optimal Density Estimation Tree (DET) can be trained on a set of
  data (specified by 'training') using cross-validation (with number of folds
  specified with the 'folds' parameter).  This trained density estimation tree
  may then be saved with the 'output_model' output parameter.
  
  The variable importances (that is, the feature importance values for each
  dimension) may be saved with the 'vi' output parameter, and the density
  estimates for each training point may be saved with the
  'training_set_estimates' output parameter.
  
  Enabling path printing for each node outputs the path from the root node to a
  leaf for each entry in the test set, or training set (if a test set is not
  provided).  Strings like 'LRLRLR' (indicating that traversal went to the left
  child, then the right child, then the left child, and so forth) will be
  output. If 'lr-id' or 'id-lr' are given as the 'path_format' parameter, then
  the ID (tag) of every node along the path will be printed after or before the
  L or R character indicating the direction of traversal, respectively.
  
  This program also can provide density estimates for a set of test points,
  specified in the 'test' parameter.  The density estimation tree used for this
  task will be the tree that was trained on the given training points, or a tree
  given as the parameter 'input_model'.  The density estimates for the test
  points may be saved using the 'test_set_estimates' output parameter.


  Input parameters:

   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - folds (int): The number of folds of cross-validation to perform for
        the estimation (0 is LOOCV)  Default value 10.
   - input_model (DTree): Trained density estimation tree to load.
   - max_leaf_size (int): The maximum size of a leaf in the unpruned,
        fully grown DET.  Default value 10.
   - min_leaf_size (int): The minimum size of a leaf in the unpruned,
        fully grown DET.  Default value 5.
   - path_format (string): The format of path printing: 'lr', 'id-lr', or
        'lr-id'.  Default value 'lr'.
   - skip_pruning (bool): Whether to bypass the pruning process and output
        the unpruned tree only.
   - test (mat.Dense): A set of test points to estimate the density of.
   - training (mat.Dense): The data set on which to build a density
        estimation tree.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output_model (DTree): Output to save trained density estimation tree
        to.
   - tag_counters_file (string): The file to output the number of points
        that went to each leaf.  Default value ''.
   - tag_file (string): The file to output the tags (and possibly paths)
        for each sample in the test set.  Default value ''.
   - test_set_estimates (mat.Dense): The output estimates on the test set
        from the final optimally pruned tree.
   - training_set_estimates (mat.Dense): The output density estimates on
        the training set from the final optimally pruned tree.
   - vi (mat.Dense): The output variable importance values for each
        feature.

*/
func Det(param *DetOptionalParam) (DTree, string, string, *mat.Dense, *mat.Dense, *mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Density Estimation With Density Estimation Trees")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Folds != 10 {
    SetParamInt("folds", param.Folds)
    SetPassed("folds")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setDTree("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max_leaf_size != 10 {
    SetParamInt("max_leaf_size", param.Max_leaf_size)
    SetPassed("max_leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Min_leaf_size != 5 {
    SetParamInt("min_leaf_size", param.Min_leaf_size)
    SetPassed("min_leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Path_format != "lr" {
    SetParamString("path_format", param.Path_format)
    SetPassed("path_format")
  }

  // Detect if the parameter was passed; set if so.
  if param.Skip_pruning != false {
    SetParamBool("skip_pruning", param.Skip_pruning)
    SetPassed("skip_pruning")
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
  SetPassed("tag_counters_file")
  SetPassed("tag_file")
  SetPassed("test_set_estimates")
  SetPassed("training_set_estimates")
  SetPassed("vi")

  // Call the mlpack program.
  C.mlpackdet()

  // Initialize result variable and get output.
  var output_model DTree
  output_model.getDTree("output_model")
  tag_counters_file := GetParamString("tag_counters_file")
  tag_file := GetParamString("tag_file")
  var test_set_estimates_ptr mlpackArma
  test_set_estimates := test_set_estimates_ptr.ArmaToGonumMat("test_set_estimates")
  var training_set_estimates_ptr mlpackArma
  training_set_estimates := training_set_estimates_ptr.ArmaToGonumMat("training_set_estimates")
  var vi_ptr mlpackArma
  vi := vi_ptr.ArmaToGonumMat("vi")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output_model, tag_counters_file, tag_file, test_set_estimates, training_set_estimates, vi
}
