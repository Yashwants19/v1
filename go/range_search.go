package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_range_search
#include <capi/range_search.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type Range_searchOptionalParam struct {
    Copy_all_inputs bool
    Input_model *RSModel
    Leaf_size int
    Max float64
    Min float64
    Naive bool
    Query *mat.Dense
    Random_basis bool
    Reference *mat.Dense
    Seed int
    Single_mode bool
    Tree_type string
    Verbose bool
}

func InitializeRange_search() *Range_searchOptionalParam {
  return &Range_searchOptionalParam{
    Copy_all_inputs: false,
    Input_model: nil,
    Leaf_size: 20,
    Max: 0,
    Min: 0,
    Naive: false,
    Query: nil,
    Random_basis: false,
    Reference: nil,
    Seed: 0,
    Single_mode: false,
    Tree_type: "kd",
    Verbose: false,
  }
}

type RSModel struct {
 mem unsafe.Pointer
}

func (m *RSModel) allocRSModel(identifier string) {
 m.mem = C.mlpackGetRSModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *RSModel) getRSModel(identifier string) {
 m.allocRSModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setRSModel(identifier string, ptr *RSModel) {
 C.mlpackSetRSModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program implements range search with a Euclidean distance metric. For a
  given query point, a given range, and a given set of reference points, the
  program will return all of the reference points with distance to the query
  point in the given range.  This is performed for an entire set of query
  points. You may specify a separate set of reference and query points, or only
  a reference set -- which is then used as both the reference and query set. 
  The given range is taken to be inclusive (that is, points with a distance
  exactly equal to the minimum and maximum of the range are included in the
  results).
  
  For example, the following will calculate the points within the range [2, 5]
  of each point in 'input.csv' and store the distances in 'distances.csv' and
  the neighbors in 'neighbors.csv':
  
  $ range_search --min=2 --max=5 --reference_file=input.csv
    --distances_file=distances.csv --neighbors_file=neighbors.csv
  
  The output files are organized such that line i corresponds to the points
  found for query point i.  Because sometimes 0 points may be found in the given
  range, lines of the output files may be empty.  The points are not ordered in
  any specific manner.
  
  Because the number of points returned for each query point may differ, the
  resultant CSV-like files may not be loadable by many programs.  However, at
  this time a better way to store this non-square result is not known.  As a
  result, any output files will be written as CSVs in this manner, regardless of
  the given extension.


  Input parameters:

   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - input_model (RSModel): File containing pre-trained range search
        model.
   - leaf_size (int): Leaf size for tree building (used for kd-trees, vp
        trees, random projection trees, UB trees, R trees, R* trees, X trees,
        Hilbert R trees, R+ trees, R++ trees, and octrees).  Default value 20.
   - max (float64): Upper bound in range (if not specified, +inf will be
        used.  Default value 0.
   - min (float64): Lower bound in range.  Default value 0.
   - naive (bool): If true, O(n^2) naive mode is used for computation.
   - query (mat.Dense): File containing query points (optional).
   - random_basis (bool): Before tree-building, project the data onto a
        random orthogonal basis.
   - reference (mat.Dense): Matrix containing the reference dataset.
   - seed (int): Random seed (if 0, std::time(NULL) is used).  Default
        value 0.
   - single_mode (bool): If true, single-tree search is used (as opposed
        to dual-tree search).
   - tree_type (string): Type of tree to use: 'kd', 'vp', 'rp', 'max-rp',
        'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus',
        'r-plus-plus', 'oct'.  Default value 'kd'.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - distances_file (string): File to output distances into.  Default
        value ''.
   - neighbors_file (string): File to output neighbors into.  Default
        value ''.
   - output_model (RSModel): If specified, the range search model will be
        saved to the given file.

*/
func Range_search(param *Range_searchOptionalParam) (string, string, RSModel) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Range Search")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setRSModel("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Leaf_size != 20 {
    SetParamInt("leaf_size", param.Leaf_size)
    SetPassed("leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max != 0 {
    SetParamDouble("max", param.Max)
    SetPassed("max")
  }

  // Detect if the parameter was passed; set if so.
  if param.Min != 0 {
    SetParamDouble("min", param.Min)
    SetPassed("min")
  }

  // Detect if the parameter was passed; set if so.
  if param.Naive != false {
    SetParamBool("naive", param.Naive)
    SetPassed("naive")
  }

  // Detect if the parameter was passed; set if so.
  if param.Query != nil {
    GonumToArmaMat("query", param.Query)
    SetPassed("query")
  }

  // Detect if the parameter was passed; set if so.
  if param.Random_basis != false {
    SetParamBool("random_basis", param.Random_basis)
    SetPassed("random_basis")
  }

  // Detect if the parameter was passed; set if so.
  if param.Reference != nil {
    GonumToArmaMat("reference", param.Reference)
    SetPassed("reference")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    SetParamInt("seed", param.Seed)
    SetPassed("seed")
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
  SetPassed("distances_file")
  SetPassed("neighbors_file")
  SetPassed("output_model")

  // Call the mlpack program.
  C.mlpackrange_search()

  // Initialize result variable and get output.
  distances_file := GetParamString("distances_file")
  neighbors_file := GetParamString("neighbors_file")
  var output_model RSModel
  output_model.getRSModel("output_model")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return distances_file, neighbors_file, output_model
}
