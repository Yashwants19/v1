package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_fastmks
#include <capi/fastmks.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type FastmksOptionalParam struct {
    Bandwidth float64
    Base float64
    Copy_all_inputs bool
    Degree float64
    Input_model *FastMKSModel
    K int
    Kernel string
    Naive bool
    Offset float64
    Query *mat.Dense
    Reference *mat.Dense
    Scale float64
    Single bool
    Verbose bool
}

func InitializeFastmks() *FastmksOptionalParam {
  return &FastmksOptionalParam{
    Bandwidth: 1,
    Base: 2,
    Copy_all_inputs: false,
    Degree: 2,
    Input_model: nil,
    K: 0,
    Kernel: "linear",
    Naive: false,
    Offset: 0,
    Query: nil,
    Reference: nil,
    Scale: 1,
    Single: false,
    Verbose: false,
  }
}

type FastMKSModel struct {
 mem unsafe.Pointer
}

func (m *FastMKSModel) allocFastMKSModel(identifier string) {
 m.mem = C.mlpackGetFastMKSModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *FastMKSModel) getFastMKSModel(identifier string) {
 m.allocFastMKSModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setFastMKSModel(identifier string, ptr *FastMKSModel) {
 C.mlpackSetFastMKSModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program will find the k maximum kernels of a set of points, using a query
  set and a reference set (which can optionally be the same set). More
  specifically, for each point in the query set, the k points in the reference
  set with maximum kernel evaluations are found.  The kernel function used is
  specified with the 'kernel' parameter.
  
  For example, the following command will calculate, for each point in the query
  set query, the five points in the reference set reference with maximum kernel
  evaluation using the linear kernel.  The kernel evaluations may be saved with
  the  kernels output parameter and the indices may be saved with the indices
  output parameter.
  
  param := InitializeFastmks()
  param.K = 5
  param.Reference = reference
  param.Query = query
  param.Kernel = "linear"
  indices, kernels, _ := Fastmks(param)
  
  The output matrices are organized such that row i and column j in the indices
  matrix corresponds to the index of the point in the reference set that has
  j'th largest kernel evaluation with the point in the query set with index i. 
  Row i and column j in the kernels matrix corresponds to the kernel evaluation
  between those two points.
  
  This program performs FastMKS using a cover tree.  The base used to build the
  cover tree can be specified with the 'base' parameter.


  Input parameters:

   - bandwidth (float64): Bandwidth (for Gaussian, Epanechnikov, and
        triangular kernels).  Default value 1.
   - base (float64): Base to use during cover tree construction.  Default
        value 2.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - degree (float64): Degree of polynomial kernel.  Default value 2.
   - input_model (FastMKSModel): Input FastMKS model to use.
   - k (int): Number of maximum kernels to find.  Default value 0.
   - kernel (string): Kernel type to use: 'linear', 'polynomial',
        'cosine', 'gaussian', 'epanechnikov', 'triangular', 'hyptan'.  Default
        value 'linear'.
   - naive (bool): If true, O(n^2) naive mode is used for computation.
   - offset (float64): Offset of kernel (for polynomial and hyptan
        kernels).  Default value 0.
   - query (mat.Dense): The query dataset.
   - reference (mat.Dense): The reference dataset.
   - scale (float64): Scale of kernel (for hyptan kernel).  Default value
        1.
   - single (bool): If true, single-tree search is used (as opposed to
        dual-tree search.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - indices (mat.Dense): Output matrix of indices.
   - kernels (mat.Dense): Output matrix of kernels.
   - output_model (FastMKSModel): Output for FastMKS model.

*/
func Fastmks(param *FastmksOptionalParam) (*mat.Dense, *mat.Dense, FastMKSModel) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("FastMKS (Fast Max-Kernel Search)")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Bandwidth != 1 {
    SetParamDouble("bandwidth", param.Bandwidth)
    SetPassed("bandwidth")
  }

  // Detect if the parameter was passed; set if so.
  if param.Base != 2 {
    SetParamDouble("base", param.Base)
    SetPassed("base")
  }

  // Detect if the parameter was passed; set if so.
  if param.Degree != 2 {
    SetParamDouble("degree", param.Degree)
    SetPassed("degree")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setFastMKSModel("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.K != 0 {
    SetParamInt("k", param.K)
    SetPassed("k")
  }

  // Detect if the parameter was passed; set if so.
  if param.Kernel != "linear" {
    SetParamString("kernel", param.Kernel)
    SetPassed("kernel")
  }

  // Detect if the parameter was passed; set if so.
  if param.Naive != false {
    SetParamBool("naive", param.Naive)
    SetPassed("naive")
  }

  // Detect if the parameter was passed; set if so.
  if param.Offset != 0 {
    SetParamDouble("offset", param.Offset)
    SetPassed("offset")
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
  if param.Scale != 1 {
    SetParamDouble("scale", param.Scale)
    SetPassed("scale")
  }

  // Detect if the parameter was passed; set if so.
  if param.Single != false {
    SetParamBool("single", param.Single)
    SetPassed("single")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("indices")
  SetPassed("kernels")
  SetPassed("output_model")

  // Call the mlpack program.
  C.mlpackfastmks()

  // Initialize result variable and get output.
  var indices_ptr mlpackArma
  indices := indices_ptr.ArmaToGonumUmat("indices")
  var kernels_ptr mlpackArma
  kernels := kernels_ptr.ArmaToGonumMat("kernels")
  var output_model FastMKSModel
  output_model.getFastMKSModel("output_model")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return indices, kernels, output_model
}
