package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_dbscan
#include <capi/dbscan.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type DbscanOptionalParam struct {
    Copy_all_inputs bool
    Epsilon float64
    Min_size int
    Naive bool
    Selection_type string
    Single_mode bool
    Tree_type string
    Verbose bool
}

func InitializeDbscan() *DbscanOptionalParam {
  return &DbscanOptionalParam{
    Copy_all_inputs: false,
    Epsilon: 1,
    Min_size: 5,
    Naive: false,
    Selection_type: "ordered",
    Single_mode: false,
    Tree_type: "kd",
    Verbose: false,
  }
}

/*
  This program implements the DBSCAN algorithm for clustering using accelerated
  tree-based range search.  The type of tree that is used may be parameterized,
  or brute-force range search may also be used.
  
  The input dataset to be clustered may be specified with the 'input' parameter;
  the radius of each range search may be specified with the 'epsilon'
  parameters, and the minimum number of points in a cluster may be specified
  with the 'min_size' parameter.
  
  The 'assignments' and 'centroids' output parameters may be used to save the
  output of the clustering. 'assignments' contains the cluster assignments of
  each point, and 'centroids' contains the centroids of each cluster.
  
  The range search may be controlled with the 'tree_type', 'single_mode', and
  'naive' parameters.  'tree_type' can control the type of tree used for range
  search; this can take a variety of values: 'kd', 'r', 'r-star', 'x',
  'hilbert-r', 'r-plus', 'r-plus-plus', 'cover', 'ball'. The 'single_mode'
  parameter will force single-tree search (as opposed to the default dual-tree
  search), and ''naive' will force brute-force range search.
  
  An example usage to run DBSCAN on the dataset in input with a radius of 0.5
  and a minimum cluster size of 5 is given below:
  
  param := InitializeDbscan()
  param.Epsilon = 0.5
  param.Min_size = 5
  _, _ := Dbscan(input, param)


  Input parameters:

   - input (mat.Dense): Input dataset to cluster.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - epsilon (float64): Radius of each range search.  Default value 1.
   - min_size (int): Minimum number of points for a cluster.  Default
        value 5.
   - naive (bool): If set, brute-force range search (not tree-based) will
        be used.
   - selection_type (string): If using point selection policy, the type of
        selection to use ('ordered', 'random').  Default value 'ordered'.
   - single_mode (bool): If set, single-tree range search (not dual-tree)
        will be used.
   - tree_type (string): If using single-tree or dual-tree search, the
        type of tree to use ('kd', 'r', 'r-star', 'x', 'hilbert-r', 'r-plus',
        'r-plus-plus', 'cover', 'ball').  Default value 'kd'.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - assignments (mat.VecDense): Output matrix for assignments of each
        point.
   - centroids (mat.Dense): Matrix to save output centroids to.

*/
func Dbscan(input *mat.Dense, param *DbscanOptionalParam) (*mat.VecDense, *mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("DBSCAN clustering")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  GonumToArmaMat("input", input)
  SetPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Epsilon != 1 {
    SetParamDouble("epsilon", param.Epsilon)
    SetPassed("epsilon")
  }

  // Detect if the parameter was passed; set if so.
  if param.Min_size != 5 {
    SetParamInt("min_size", param.Min_size)
    SetPassed("min_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Naive != false {
    SetParamBool("naive", param.Naive)
    SetPassed("naive")
  }

  // Detect if the parameter was passed; set if so.
  if param.Selection_type != "ordered" {
    SetParamString("selection_type", param.Selection_type)
    SetPassed("selection_type")
  }

  // Detect if the parameter was passed; set if so.
  if param.Single_mode != false {
    SetParamBool("single_mode", param.Single_mode)
    SetPassed("single_mode")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tree_type != "kd" {
    SetParamString("tree_type", param.Tree_type)
    SetPassed("tree_type")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("assignments")
  SetPassed("centroids")

  // Call the mlpack program.
  C.mlpackdbscan()

  // Initialize result variable and get output.
  var assignments_ptr mlpackArma
  assignments := assignments_ptr.ArmaToGonumUrow("assignments")
  var centroids_ptr mlpackArma
  centroids := centroids_ptr.ArmaToGonumMat("centroids")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return assignments, centroids
}
