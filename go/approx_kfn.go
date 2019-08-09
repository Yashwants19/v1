package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_approx_kfn
#include <capi/approx_kfn.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type Approx_kfnOptionalParam struct {
    Algorithm string
    Calculate_error bool
    Copy_all_inputs bool
    Exact_distances *mat.Dense
    Input_model *ApproxKFNModel
    K int
    Num_projections int
    Num_tables int
    Query *mat.Dense
    Reference *mat.Dense
    Verbose bool
}

func InitializeApprox_kfn() *Approx_kfnOptionalParam {
  return &Approx_kfnOptionalParam{
    Algorithm: "ds",
    Calculate_error: false,
    Copy_all_inputs: false,
    Exact_distances: nil,
    Input_model: nil,
    K: 0,
    Num_projections: 5,
    Num_tables: 5,
    Query: nil,
    Reference: nil,
    Verbose: false,
  }
}

type ApproxKFNModel struct {
 mem unsafe.Pointer
}

func (m *ApproxKFNModel) allocApproxKFNModel(identifier string) {
 m.mem = C.mlpackGetApproxKFNModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *ApproxKFNModel) getApproxKFNModel(identifier string) {
 m.allocApproxKFNModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setApproxKFNModel(identifier string, ptr *ApproxKFNModel) {
 C.mlpackSetApproxKFNModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program implements two strategies for furthest neighbor search. These
  strategies are:
  
   - The 'qdafn' algorithm from "Approximate Furthest Neighbor in High
  Dimensions" by R. Pagh, F. Silvestri, J. Sivertsen, and M. Skala, in
  Similarity Search and Applications 2015 (SISAP).
   - The 'DrusillaSelect' algorithm from "Fast approximate furthest neighbors
  with data-dependent candidate selection", by R.R. Curtin and A.B. Gardner, in
  Similarity Search and Applications 2016 (SISAP).
  
  These two strategies give approximate results for the furthest neighbor search
  problem and can be used as fast replacements for other furthest neighbor
  techniques such as those found in the mlpack_kfn program.  Note that
  typically, the 'ds' algorithm requires far fewer tables and projections than
  the 'qdafn' algorithm.
  
  Specify a reference set (set to search in) with 'reference', specify a query
  set with 'query', and specify algorithm parameters with 'num_tables' and
  'num_projections' (or don't and defaults will be used).  The algorithm to be
  used (either 'ds'---the default---or 'qdafn')  may be specified with
  'algorithm'.  Also specify the number of neighbors to search for with 'k'.
  
  If no query set is specified, the reference set will be used as the query set.
   The 'output_model' output parameter may be used to store the built model, and
  an input model may be loaded instead of specifying a reference set with the
  'input_model' option.
  
  Results for each query point can be stored with the 'neighbors' and
  'distances' output parameters.  Each row of these output matrices holds the k
  distances or neighbor indices for each query point.
  
  For example, to find the 5 approximate furthest neighbors with reference_set
  as the reference set and query_set as the query set using DrusillaSelect,
  storing the furthest neighbor indices to neighbors and the furthest neighbor
  distances to distances, one could call
  
  param := InitializeApprox_kfn()
  param.Query = query_set
  param.Reference = reference_set
  param.K = 5
  param.Algorithm = "ds"
  distances, neighbors, _ := Approx_kfn(param)
  
  and to perform approximate all-furthest-neighbors search with k=1 on the set
  data storing only the furthest neighbor distances to distances, one could call
  
  param := InitializeApprox_kfn()
  param.Reference = reference_set
  param.K = 1
  distances, _, _ := Approx_kfn(param)
  
  A trained model can be re-used.  If a model has been previously saved to
  model, then we may find 3 approximate furthest neighbors on a query set
  new_query_set using that model and store the furthest neighbor indices into
  neighbors by calling
  
  param := InitializeApprox_kfn()
  param.Input_model = model
  param.Query = new_query_set
  param.K = 3
  _, neighbors, _ := Approx_kfn(param)


  Input parameters:

   - algorithm (string): Algorithm to use: 'ds' or 'qdafn'.  Default value
        'ds'.
   - calculate_error (bool): If set, calculate the average distance error
        for the first furthest neighbor only.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - exact_distances (mat.Dense): Matrix containing exact distances to
        furthest neighbors; this can be used to avoid explicit calculation when
        --calculate_error is set.
   - input_model (ApproxKFNModel): File containing input model.
   - k (int): Number of furthest neighbors to search for.  Default value
        0.
   - num_projections (int): Number of projections to use in each hash
        table.  Default value 5.
   - num_tables (int): Number of hash tables to use.  Default value 5.
   - query (mat.Dense): Matrix containing query points.
   - reference (mat.Dense): Matrix containing the reference dataset.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - distances (mat.Dense): Matrix to save furthest neighbor distances
        to.
   - neighbors (mat.Dense): Matrix to save neighbor indices to.
   - output_model (ApproxKFNModel): File to save output model to.

*/
func Approx_kfn(param *Approx_kfnOptionalParam) (*mat.Dense, *mat.Dense, ApproxKFNModel) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Approximate furthest neighbor search")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Algorithm != "ds" {
    SetParamString("algorithm", param.Algorithm)
    SetPassed("algorithm")
  }

  // Detect if the parameter was passed; set if so.
  if param.Calculate_error != false {
    SetParamBool("calculate_error", param.Calculate_error)
    SetPassed("calculate_error")
  }

  // Detect if the parameter was passed; set if so.
  if param.Exact_distances != nil {
    GonumToArmaMat("exact_distances", param.Exact_distances)
    SetPassed("exact_distances")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setApproxKFNModel("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.K != 0 {
    SetParamInt("k", param.K)
    SetPassed("k")
  }

  // Detect if the parameter was passed; set if so.
  if param.Num_projections != 5 {
    SetParamInt("num_projections", param.Num_projections)
    SetPassed("num_projections")
  }

  // Detect if the parameter was passed; set if so.
  if param.Num_tables != 5 {
    SetParamInt("num_tables", param.Num_tables)
    SetPassed("num_tables")
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
  C.mlpackapprox_kfn()

  // Initialize result variable and get output.
  var distances_ptr mlpackArma
  distances := distances_ptr.ArmaToGonumMat("distances")
  var neighbors_ptr mlpackArma
  neighbors := neighbors_ptr.ArmaToGonumUmat("neighbors")
  var output_model ApproxKFNModel
  output_model.getApproxKFNModel("output_model")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return distances, neighbors, output_model
}
