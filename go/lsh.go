package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_lsh
#include <capi/lsh.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type LshOptionalParam struct {
    Bucket_size int
    Copy_all_inputs bool
    Hash_width float64
    Input_model *LSHSearch
    K int
    Num_probes int
    Projections int
    Query *mat.Dense
    Reference *mat.Dense
    Second_hash_size int
    Seed int
    Tables int
    True_neighbors *mat.Dense
    Verbose bool
}

func InitializeLsh() *LshOptionalParam {
  return &LshOptionalParam{
    Bucket_size: 500,
    Copy_all_inputs: false,
    Hash_width: 0,
    Input_model: nil,
    K: 0,
    Num_probes: 0,
    Projections: 10,
    Query: nil,
    Reference: nil,
    Second_hash_size: 99901,
    Seed: 0,
    Tables: 30,
    True_neighbors: nil,
    Verbose: false,
  }
}

type LSHSearch struct {
 mem unsafe.Pointer
}

func (m *LSHSearch) allocLSHSearch(identifier string) {
 m.mem = C.mlpackGetLSHSearchPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *LSHSearch) getLSHSearch(identifier string) {
 m.allocLSHSearch(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setLSHSearch(identifier string, ptr *LSHSearch) {
 C.mlpackSetLSHSearchPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program will calculate the k approximate-nearest-neighbors of a set of
  points using locality-sensitive hashing. You may specify a separate set of
  reference points and query points, or just a reference set which will be used
  as both the reference and query set. 
  
  For example, the following will return 5 neighbors from the data for each
  point in input and store the distances in distances and the neighbors in
  neighbors:
  
  param := InitializeLsh()
  param.K = 5
  param.Reference = input
  distances, neighbors, _ := Lsh(param)
  
  The output is organized such that row i and column j in the neighbors output
  corresponds to the index of the point in the reference set which is the j'th
  nearest neighbor from the point in the query set with index i.  Row j and
  column i in the distances output file corresponds to the distance between
  those two points.
  
  Because this is approximate-nearest-neighbors search, results may be different
  from run to run.  Thus, the 'seed' parameter can be specified to set the
  random seed.
  
  This program also has many other parameters to control its functionality; see
  the parameter-specific documentation for more information.


  Input parameters:

   - bucket_size (int): The size of a bucket in the second level hash. 
        Default value 500.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - hash_width (float64): The hash width for the first-level hashing in
        the LSH preprocessing. By default, the LSH class automatically estimates
        a hash width for its use.  Default value 0.
   - input_model (LSHSearch): Input LSH model.
   - k (int): Number of nearest neighbors to find.  Default value 0.
   - num_probes (int): Number of additional probes for multiprobe LSH; if
        0, traditional LSH is used.  Default value 0.
   - projections (int): The number of hash functions for each table 
        Default value 10.
   - query (mat.Dense): Matrix containing query points (optional).
   - reference (mat.Dense): Matrix containing the reference dataset.
   - second_hash_size (int): The size of the second level hash table. 
        Default value 99901.
   - seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - tables (int): The number of hash tables to be used.  Default value
        30.
   - true_neighbors (mat.Dense): Matrix of true neighbors to compute
        recall with (the recall is printed when -v is specified).
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - distances (mat.Dense): Matrix to output distances into.
   - neighbors (mat.Dense): Matrix to output neighbors into.
   - output_model (LSHSearch): Output for trained LSH model.

*/
func Lsh(param *LshOptionalParam) (*mat.Dense, *mat.Dense, LSHSearch) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("K-Approximate-Nearest-Neighbor Search with LSH")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Bucket_size != 500 {
    SetParamInt("bucket_size", param.Bucket_size)
    SetPassed("bucket_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Hash_width != 0 {
    SetParamDouble("hash_width", param.Hash_width)
    SetPassed("hash_width")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setLSHSearch("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.K != 0 {
    SetParamInt("k", param.K)
    SetPassed("k")
  }

  // Detect if the parameter was passed; set if so.
  if param.Num_probes != 0 {
    SetParamInt("num_probes", param.Num_probes)
    SetPassed("num_probes")
  }

  // Detect if the parameter was passed; set if so.
  if param.Projections != 10 {
    SetParamInt("projections", param.Projections)
    SetPassed("projections")
  }

  // Detect if the parameter was passed; set if so.
  if param.Query != nil {
    GonumToArmaMat("query", param.Query)
    SetPassed("query")
  }

  // Detect if the parameter was passed; set if so.
  if param.Reference != nil {
    GonumToArmaMat("reference", param.Reference)
    SetPassed("reference")
  }

  // Detect if the parameter was passed; set if so.
  if param.Second_hash_size != 99901 {
    SetParamInt("second_hash_size", param.Second_hash_size)
    SetPassed("second_hash_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    SetParamInt("seed", param.Seed)
    SetPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tables != 30 {
    SetParamInt("tables", param.Tables)
    SetPassed("tables")
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
  C.mlpacklsh()

  // Initialize result variable and get output.
  var distances_ptr mlpackArma
  distances := distances_ptr.ArmaToGonumMat("distances")
  var neighbors_ptr mlpackArma
  neighbors := neighbors_ptr.ArmaToGonumUmat("neighbors")
  var output_model LSHSearch
  output_model.getLSHSearch("output_model")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return distances, neighbors, output_model
}
