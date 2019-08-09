package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_kfn
#include <capi/kfn.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type KfnOptionalParam struct {
    Algorithm string
    Copy_all_inputs bool
    Epsilon float64
    Input_model *KFNModel
    K int
    Leaf_size int
    Percentage float64
    Query *mat.Dense
    Random_basis bool
    Reference *mat.Dense
    Seed int
    Tree_type string
    True_distances *mat.Dense
    True_neighbors *mat.Dense
    Verbose bool
}

func InitializeKfn() *KfnOptionalParam {
  return &KfnOptionalParam{
    Algorithm: "dual_tree",
    Copy_all_inputs: false,
    Epsilon: 0,
    Input_model: nil,
    K: 0,
    Leaf_size: 20,
    Percentage: 1,
    Query: nil,
    Random_basis: false,
    Reference: nil,
    Seed: 0,
    Tree_type: "kd",
    True_distances: nil,
    True_neighbors: nil,
    Verbose: false,
  }
}

type KFNModel struct {
 mem unsafe.Pointer
}

func (m *KFNModel) allocKFNModel(identifier string) {
 m.mem = C.mlpackGetKFNModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *KFNModel) getKFNModel(identifier string) {
 m.allocKFNModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setKFNModel(identifier string, ptr *KFNModel) {
 C.mlpackSetKFNModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program will calculate the k-furthest-neighbors of a set of points. You
  may specify a separate set of reference points and query points, or just a
  reference set which will be used as both the reference and query set.
  
  For example, the following will calculate the 5 furthest neighbors of
  eachpoint in input and store the distances in distances and the neighbors in
  neighbors: 
  
  param := InitializeKfn()
  param.K = 5
  param.Reference = input
  distances, neighbors, _ := Kfn(param)
  
  The output files are organized such that row i and column j in the neighbors
  output matrix corresponds to the index of the point in the reference set which
  is the j'th furthest neighbor from the point in the query set with index i. 
  Row i and column j in the distances output file corresponds to the distance
  between those two points.


  Input parameters:

   - algorithm (string): Type of neighbor search: 'naive', 'single_tree',
        'dual_tree', 'greedy'.  Default value 'dual_tree'.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - epsilon (float64): If specified, will do approximate furthest
        neighbor search with given relative error. Must be in the range [0,1). 
        Default value 0.
   - input_model (KFNModel): Pre-trained kFN model.
   - k (int): Number of furthest neighbors to find.  Default value 0.
   - leaf_size (int): Leaf size for tree building (used for kd-trees, vp
        trees, random projection trees, UB trees, R trees, R* trees, X trees,
        Hilbert R trees, R+ trees, R++ trees, and octrees).  Default value 20.
   - percentage (float64): If specified, will do approximate furthest
        neighbor search. Must be in the range (0,1] (decimal form). Resultant
        neighbors will be at least (p*100) % of the distance as the true
        furthest neighbor.  Default value 1.
   - query (mat.Dense): Matrix containing query points (optional).
   - random_basis (bool): Before tree-building, project the data onto a
        random orthogonal basis.
   - reference (mat.Dense): Matrix containing the reference dataset.
   - seed (int): Random seed (if 0, std::time(NULL) is used).  Default
        value 0.
   - tree_type (string): Type of tree to use: 'kd', 'vp', 'rp', 'max-rp',
        'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus',
        'r-plus-plus', 'oct'.  Default value 'kd'.
   - true_distances (mat.Dense): Matrix of true distances to compute the
        effective error (average relative error) (it is printed when -v is
        specified).
   - true_neighbors (mat.Dense): Matrix of true neighbors to compute the
        recall (it is printed when -v is specified).
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - distances (mat.Dense): Matrix to output distances into.
   - neighbors (mat.Dense): Matrix to output neighbors into.
   - output_model (KFNModel): If specified, the kFN model will be output
        here.

*/
func Kfn(param *KfnOptionalParam) (*mat.Dense, *mat.Dense, KFNModel) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("k-Furthest-Neighbors Search")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Algorithm != "dual_tree" {
    SetParamString("algorithm", param.Algorithm)
    SetPassed("algorithm")
  }

  // Detect if the parameter was passed; set if so.
  if param.Epsilon != 0 {
    SetParamDouble("epsilon", param.Epsilon)
    SetPassed("epsilon")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setKFNModel("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.K != 0 {
    SetParamInt("k", param.K)
    SetPassed("k")
  }

  // Detect if the parameter was passed; set if so.
  if param.Leaf_size != 20 {
    SetParamInt("leaf_size", param.Leaf_size)
    SetPassed("leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Percentage != 1 {
    SetParamDouble("percentage", param.Percentage)
    SetPassed("percentage")
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
  if param.Tree_type != "kd" {
    SetParamString("tree_type", param.Tree_type)
    SetPassed("tree_type")
  }

  // Detect if the parameter was passed; set if so.
  if param.True_distances != nil {
    GonumToArmaMat("true_distances", param.True_distances)
    SetPassed("true_distances")
  }

  // Detect if the parameter was passed; set if so.
  if param.True_neighbors != nil {
    GonumToArmaUmat("true_neighbors", param.True_neighbors)
    SetPassed("true_neighbors")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("distances")
  SetPassed("neighbors")
  SetPassed("output_model")

  // Call the mlpack program.
  C.mlpackkfn()

  // Initialize result variable and get output.
  var distances_ptr mlpackArma
  distances := distances_ptr.ArmaToGonumMat("distances")
  var neighbors_ptr mlpackArma
  neighbors := neighbors_ptr.ArmaToGonumUmat("neighbors")
  var output_model KFNModel
  output_model.getKFNModel("output_model")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return distances, neighbors, output_model
}
