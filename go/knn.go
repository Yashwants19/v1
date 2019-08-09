package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_knn
#include <capi/knn.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type KnnOptionalParam struct {
    Algorithm string
    Copy_all_inputs bool
    Epsilon float64
    Input_model *KNNModel
    K int
    Leaf_size int
    Query *mat.Dense
    Random_basis bool
    Reference *mat.Dense
    Rho float64
    Seed int
    Tau float64
    Tree_type string
    True_distances *mat.Dense
    True_neighbors *mat.Dense
    Verbose bool
}

func InitializeKnn() *KnnOptionalParam {
  return &KnnOptionalParam{
    Algorithm: "dual_tree",
    Copy_all_inputs: false,
    Epsilon: 0,
    Input_model: nil,
    K: 0,
    Leaf_size: 20,
    Query: nil,
    Random_basis: false,
    Reference: nil,
    Rho: 0.7,
    Seed: 0,
    Tau: 0,
    Tree_type: "kd",
    True_distances: nil,
    True_neighbors: nil,
    Verbose: false,
  }
}

type KNNModel struct {
 mem unsafe.Pointer
}

func (m *KNNModel) allocKNNModel(identifier string) {
 m.mem = C.mlpackGetKNNModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *KNNModel) getKNNModel(identifier string) {
 m.allocKNNModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setKNNModel(identifier string, ptr *KNNModel) {
 C.mlpackSetKNNModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program will calculate the k-nearest-neighbors of a set of points using
  kd-trees or cover trees (cover tree support is experimental and may be slow).
  You may specify a separate set of reference points and query points, or just a
  reference set which will be used as both the reference and query set.
  
  For example, the following command will calculate the 5 nearest neighbors of
  each point in input and store the distances in distances and the neighbors in
  neighbors: 
  
  param := InitializeKnn()
  param.K = 5
  param.Reference = input
  _, neighbors, _ := Knn(param)
  
  The output is organized such that row i and column j in the neighbors output
  matrix corresponds to the index of the point in the reference set which is the
  j'th nearest neighbor from the point in the query set with index i.  Row j and
  column i in the distances output matrix corresponds to the distance between
  those two points.


  Input parameters:

   - algorithm (string): Type of neighbor search: 'naive', 'single_tree',
        'dual_tree', 'greedy'.  Default value 'dual_tree'.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - epsilon (float64): If specified, will do approximate nearest neighbor
        search with given relative error.  Default value 0.
   - input_model (KNNModel): Pre-trained kNN model.
   - k (int): Number of nearest neighbors to find.  Default value 0.
   - leaf_size (int): Leaf size for tree building (used for kd-trees, vp
        trees, random projection trees, UB trees, R trees, R* trees, X trees,
        Hilbert R trees, R+ trees, R++ trees, spill trees, and octrees). 
        Default value 20.
   - query (mat.Dense): Matrix containing query points (optional).
   - random_basis (bool): Before tree-building, project the data onto a
        random orthogonal basis.
   - reference (mat.Dense): Matrix containing the reference dataset.
   - rho (float64): Balance threshold (only valid for spill trees). 
        Default value 0.7.
   - seed (int): Random seed (if 0, std::time(NULL) is used).  Default
        value 0.
   - tau (float64): Overlapping size (only valid for spill trees). 
        Default value 0.
   - tree_type (string): Type of tree to use: 'kd', 'vp', 'rp', 'max-rp',
        'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus',
        'r-plus-plus', 'spill', 'oct'.  Default value 'kd'.
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
   - output_model (KNNModel): If specified, the kNN model will be output
        here.

*/
func Knn(param *KnnOptionalParam) (*mat.Dense, *mat.Dense, KNNModel) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("k-Nearest-Neighbors Search")

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
    setKNNModel("input_model", param.Input_model)
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
  if param.Rho != 0.7 {
    SetParamDouble("rho", param.Rho)
    SetPassed("rho")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    SetParamInt("seed", param.Seed)
    SetPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tau != 0 {
    SetParamDouble("tau", param.Tau)
    SetPassed("tau")
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
  C.mlpackknn()

  // Initialize result variable and get output.
  var distances_ptr mlpackArma
  distances := distances_ptr.ArmaToGonumMat("distances")
  var neighbors_ptr mlpackArma
  neighbors := neighbors_ptr.ArmaToGonumUmat("neighbors")
  var output_model KNNModel
  output_model.getKNNModel("output_model")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return distances, neighbors, output_model
}
