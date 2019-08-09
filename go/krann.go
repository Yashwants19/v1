package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_krann
#include <capi/krann.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type KrannOptionalParam struct {
    Alpha float64
    Copy_all_inputs bool
    First_leaf_exact bool
    Input_model *RANNModel
    K int
    Leaf_size int
    Naive bool
    Query *mat.Dense
    Random_basis bool
    Reference *mat.Dense
    Sample_at_leaves bool
    Seed int
    Single_mode bool
    Single_sample_limit int
    Tau float64
    Tree_type string
    Verbose bool
}

func InitializeKrann() *KrannOptionalParam {
  return &KrannOptionalParam{
    Alpha: 0.95,
    Copy_all_inputs: false,
    First_leaf_exact: false,
    Input_model: nil,
    K: 0,
    Leaf_size: 20,
    Naive: false,
    Query: nil,
    Random_basis: false,
    Reference: nil,
    Sample_at_leaves: false,
    Seed: 0,
    Single_mode: false,
    Single_sample_limit: 20,
    Tau: 5,
    Tree_type: "kd",
    Verbose: false,
  }
}

type RANNModel struct {
 mem unsafe.Pointer
}

func (m *RANNModel) allocRANNModel(identifier string) {
 m.mem = C.mlpackGetRANNModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *RANNModel) getRANNModel(identifier string) {
 m.allocRANNModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setRANNModel(identifier string, ptr *RANNModel) {
 C.mlpackSetRANNModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program will calculate the k rank-approximate-nearest-neighbors of a set
  of points. You may specify a separate set of reference points and query
  points, or just a reference set which will be used as both the reference and
  query set. You must specify the rank approximation (in %) (and optionally the
  success probability).
  
  For example, the following will return 5 neighbors from the top 0.1% of the
  data (with probability 0.95) for each point in input and store the distances
  in distances and the neighbors in neighbors.csv:
  
  param := InitializeKrann()
  param.Reference = input
  param.K = 5
  param.Tau = 0.1
  distances, neighbors, _ := Krann(param)
  
  Note that tau must be set such that the number of points in the corresponding
  percentile of the data is greater than k.  Thus, if we choose tau = 0.1 with a
  dataset of 1000 points and k = 5, then we are attempting to choose 5 nearest
  neighbors out of the closest 1 point -- this is invalid and the program will
  terminate with an error message.
  
  The output matrices are organized such that row i and column j in the
  neighbors output file corresponds to the index of the point in the reference
  set which is the i'th nearest neighbor from the point in the query set with
  index j.  Row i and column j in the distances output file corresponds to the
  distance between those two points.


  Input parameters:

   - alpha (float64): The desired success probability.  Default value
        0.95.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - first_leaf_exact (bool): The flag to trigger sampling only after
        exactly exploring the first leaf.
   - input_model (RANNModel): Pre-trained kNN model.
   - k (int): Number of nearest neighbors to find.  Default value 0.
   - leaf_size (int): Leaf size for tree building (used for kd-trees, UB
        trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees,
        and octrees).  Default value 20.
   - naive (bool): If true, sampling will be done without using a tree.
   - query (mat.Dense): Matrix containing query points (optional).
   - random_basis (bool): Before tree-building, project the data onto a
        random orthogonal basis.
   - reference (mat.Dense): Matrix containing the reference dataset.
   - sample_at_leaves (bool): The flag to trigger sampling at leaves.
   - seed (int): Random seed (if 0, std::time(NULL) is used).  Default
        value 0.
   - single_mode (bool): If true, single-tree search is used (as opposed
        to dual-tree search.
   - single_sample_limit (int): The limit on the maximum number of samples
        (and hence the largest node you can approximate).  Default value 20.
   - tau (float64): The allowed rank-error in terms of the percentile of
        the data.  Default value 5.
   - tree_type (string): Type of tree to use: 'kd', 'ub', 'cover', 'r',
        'x', 'r-star', 'hilbert-r', 'r-plus', 'r-plus-plus', 'oct'.  Default
        value 'kd'.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - distances (mat.Dense): Matrix to output distances into.
   - neighbors (mat.Dense): Matrix to output neighbors into.
   - output_model (RANNModel): If specified, the kNN model will be output
        here.

*/
func Krann(param *KrannOptionalParam) (*mat.Dense, *mat.Dense, RANNModel) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("K-Rank-Approximate-Nearest-Neighbors (kRANN)")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Alpha != 0.95 {
    SetParamDouble("alpha", param.Alpha)
    SetPassed("alpha")
  }

  // Detect if the parameter was passed; set if so.
  if param.First_leaf_exact != false {
    SetParamBool("first_leaf_exact", param.First_leaf_exact)
    SetPassed("first_leaf_exact")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setRANNModel("input_model", param.Input_model)
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
  if param.Sample_at_leaves != false {
    SetParamBool("sample_at_leaves", param.Sample_at_leaves)
    SetPassed("sample_at_leaves")
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
  if param.Single_sample_limit != 20 {
    SetParamInt("single_sample_limit", param.Single_sample_limit)
    SetPassed("single_sample_limit")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tau != 5 {
    SetParamDouble("tau", param.Tau)
    SetPassed("tau")
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
  SetPassed("distances")
  SetPassed("neighbors")
  SetPassed("output_model")

  // Call the mlpack program.
  C.mlpackkrann()

  // Initialize result variable and get output.
  var distances_ptr mlpackArma
  distances := distances_ptr.ArmaToGonumMat("distances")
  var neighbors_ptr mlpackArma
  neighbors := neighbors_ptr.ArmaToGonumUmat("neighbors")
  var output_model RANNModel
  output_model.getRANNModel("output_model")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return distances, neighbors, output_model
}
