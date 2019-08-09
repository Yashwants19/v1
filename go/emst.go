package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_emst
#include <capi/emst.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type EmstOptionalParam struct {
    Copy_all_inputs bool
    Leaf_size int
    Naive bool
    Verbose bool
}

func InitializeEmst() *EmstOptionalParam {
  return &EmstOptionalParam{
    Copy_all_inputs: false,
    Leaf_size: 1,
    Naive: false,
    Verbose: false,
  }
}

/*
  This program can compute the Euclidean minimum spanning tree of a set of input
  points using the dual-tree Boruvka algorithm.
  
  The set to calculate the minimum spanning tree of is specified with the
  'input' parameter, and the output may be saved with the 'output' output
  parameter.
  
  The 'leaf_size' parameter controls the leaf size of the kd-tree that is used
  to calculate the minimum spanning tree, and if the 'naive' option is given,
  then brute-force search is used (this is typically much slower in low
  dimensions).  The leaf size does not affect the results, but it may have some
  effect on the runtime of the algorithm.
  
  For example, the minimum spanning tree of the input dataset data can be
  calculated with a leaf size of 20 and stored as spanning_tree using the
  following command:
  
  param := InitializeEmst()
  param.Leaf_size = 20
  spanning_tree := Emst(data, param)
  
  The output matrix is a three-dimensional matrix, where each row indicates an
  edge.  The first dimension corresponds to the lesser index of the edge; the
  second dimension corresponds to the greater index of the edge; and the third
  column corresponds to the distance between the two points.


  Input parameters:

   - input (mat.Dense): Input data matrix.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - leaf_size (int): Leaf size in the kd-tree.  One-element leaves give
        the empirically best performance, but at the cost of greater memory
        requirements.  Default value 1.
   - naive (bool): Compute the MST using O(n^2) naive algorithm.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Output data.  Stored as an edge list.

*/
func Emst(input *mat.Dense, param *EmstOptionalParam) (*mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Fast Euclidean Minimum Spanning Tree")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  GonumToArmaMat("input", input)
  SetPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Leaf_size != 1 {
    SetParamInt("leaf_size", param.Leaf_size)
    SetPassed("leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Naive != false {
    SetParamBool("naive", param.Naive)
    SetPassed("naive")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("output")

  // Call the mlpack program.
  C.mlpackemst()

  // Initialize result variable and get output.
  var output_ptr mlpackArma
  output := output_ptr.ArmaToGonumMat("output")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output
}
