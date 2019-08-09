package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_test_go_binding
#include <capi/test_go_binding.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type Test_go_bindingOptionalParam struct {
    Build_model bool
    Col_in *mat.VecDense
    Copy_all_inputs bool
    Flag1 bool
    Flag2 bool
    Matrix_and_info_in *DataWithInfo
    Matrix_in *mat.Dense
    Model_in *GaussianKernel
    Row_in *mat.VecDense
    Str_vector_in []string
    Ucol_in *mat.VecDense
    Umatrix_in *mat.Dense
    Urow_in *mat.VecDense
    Vector_in []int
    Verbose bool
}

func InitializeTest_go_binding() *Test_go_bindingOptionalParam {
  return &Test_go_bindingOptionalParam{
    Build_model: false,
    Col_in: nil,
    Copy_all_inputs: false,
    Flag1: false,
    Flag2: false,
    Matrix_and_info_in: nil,
    Matrix_in: nil,
    Model_in: nil,
    Row_in: nil,
    Ucol_in: nil,
    Umatrix_in: nil,
    Urow_in: nil,
    Verbose: false,
  }
}

type GaussianKernel struct {
 mem unsafe.Pointer
}

func (m *GaussianKernel) allocGaussianKernel(identifier string) {
 m.mem = C.mlpackGetGaussianKernelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *GaussianKernel) getGaussianKernel(identifier string) {
 m.allocGaussianKernel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setGaussianKernel(identifier string, ptr *GaussianKernel) {
 C.mlpackSetGaussianKernelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  A simple program to test Golang binding functionality.  You can build mlpack
  with the BUILD_TESTS option set to off, and this binding will no longer be
  built.


  Input parameters:

   - double_in (float64): Input double, must be 4.0.
   - int_in (int): Input int, must be 12.
   - string_in (string): Input string, must be 'hello'.
   - build_model (bool): If true, a model will be returned.
   - col_in (mat.VecDense): Input column.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - flag1 (bool): Input flag, must be specified.
   - flag2 (bool): Input flag, must not be specified.
   - matrix_and_info_in (DataWithInfo): Input matrix and info.
   - matrix_in (mat.Dense): Input matrix.
   - model_in (GaussianKernel): Input model.
   - row_in (mat.VecDense): Input row.
   - str_vector_in ([]string): Input vector of strings.
   - ucol_in (mat.VecDense): Input unsigned column.
   - umatrix_in (mat.Dense): Input unsigned matrix.
   - urow_in (mat.VecDense): Input unsigned row.
   - vector_in ([]int): Input vector of numbers.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - col_out (mat.VecDense): Output column. 2x input column
   - double_out (float64): Output double, will be 5.0.  Default value 0.
   - int_out (int): Output int, will be 13.  Default value 0.
   - matrix_and_info_out (mat.Dense): Output matrix and info; all numeric
        elements multiplied by 3.
   - matrix_out (mat.Dense): Output matrix.
   - model_bw_out (float64): The bandwidth of the model.  Default value
        0.
   - model_out (GaussianKernel): Output model, with twice the bandwidth.
   - row_out (mat.VecDense): Output row.  2x input row.
   - str_vector_out ([]string): Output string vector.
   - string_out (string): Output string, will be 'hello2'.  Default value
        ''.
   - ucol_out (mat.VecDense): Output unsigned column. 2x input column.
   - umatrix_out (mat.Dense): Output unsigned matrix.
   - urow_out (mat.VecDense): Output unsigned row.  2x input row.
   - vector_out ([]int): Output vector.

*/
func Test_go_binding(double_in float64, int_in int, string_in string, param *Test_go_bindingOptionalParam) (*mat.VecDense, float64, int, *mat.Dense, *mat.Dense, float64, GaussianKernel, *mat.VecDense, []string, string, *mat.VecDense, *mat.Dense, *mat.VecDense, []int) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Golang binding test")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  SetParamDouble("double_in", double_in)
  SetPassed("double_in")

  // Detect if the parameter was passed; set if so.
  SetParamInt("int_in", int_in)
  SetPassed("int_in")

  // Detect if the parameter was passed; set if so.
  SetParamString("string_in", string_in)
  SetPassed("string_in")

  // Detect if the parameter was passed; set if so.
  if param.Build_model != false {
    SetParamBool("build_model", param.Build_model)
    SetPassed("build_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Col_in != nil {
    GonumToArmaCol("col_in", param.Col_in)
    SetPassed("col_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.Flag1 != false {
    SetParamBool("flag1", param.Flag1)
    SetPassed("flag1")
  }

  // Detect if the parameter was passed; set if so.
  if param.Flag2 != false {
    SetParamBool("flag2", param.Flag2)
    SetPassed("flag2")
  }

  // Detect if the parameter was passed; set if so.
  if param.Matrix_and_info_in != nil {
    GonumToArmaMatWithInfo("matrix_and_info_in", param.Matrix_and_info_in)
    SetPassed("matrix_and_info_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.Matrix_in != nil {
    GonumToArmaMat("matrix_in", param.Matrix_in)
    SetPassed("matrix_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.Model_in != nil {
    setGaussianKernel("model_in", param.Model_in)
    SetPassed("model_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.Row_in != nil {
    GonumToArmaRow("row_in", param.Row_in)
    SetPassed("row_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.Str_vector_in != nil {
    SetParamVecString("str_vector_in", param.Str_vector_in)
    SetPassed("str_vector_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.Ucol_in != nil {
    GonumToArmaUcol("ucol_in", param.Ucol_in)
    SetPassed("ucol_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.Umatrix_in != nil {
    GonumToArmaUmat("umatrix_in", param.Umatrix_in)
    SetPassed("umatrix_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.Urow_in != nil {
    GonumToArmaUrow("urow_in", param.Urow_in)
    SetPassed("urow_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.Vector_in != nil {
    SetParamVecInt("vector_in", param.Vector_in)
    SetPassed("vector_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("col_out")
  SetPassed("double_out")
  SetPassed("int_out")
  SetPassed("matrix_and_info_out")
  SetPassed("matrix_out")
  SetPassed("model_bw_out")
  SetPassed("model_out")
  SetPassed("row_out")
  SetPassed("str_vector_out")
  SetPassed("string_out")
  SetPassed("ucol_out")
  SetPassed("umatrix_out")
  SetPassed("urow_out")
  SetPassed("vector_out")

  // Call the mlpack program.
  C.mlpacktest_go_binding()

  // Initialize result variable and get output.
  var col_out_ptr mlpackArma
  col_out := col_out_ptr.ArmaToGonumCol("col_out")
  double_out := GetParamDouble("double_out")
  int_out := GetParamInt("int_out")
  var matrix_and_info_out_ptr mlpackArma
  matrix_and_info_out := matrix_and_info_out_ptr.ArmaToGonumMat("matrix_and_info_out")
  var matrix_out_ptr mlpackArma
  matrix_out := matrix_out_ptr.ArmaToGonumMat("matrix_out")
  model_bw_out := GetParamDouble("model_bw_out")
  var model_out GaussianKernel
  model_out.getGaussianKernel("model_out")
  var row_out_ptr mlpackArma
  row_out := row_out_ptr.ArmaToGonumRow("row_out")
  str_vector_out := GetParamVecString("str_vector_out")
  string_out := GetParamString("string_out")
  var ucol_out_ptr mlpackArma
  ucol_out := ucol_out_ptr.ArmaToGonumUcol("ucol_out")
  var umatrix_out_ptr mlpackArma
  umatrix_out := umatrix_out_ptr.ArmaToGonumUmat("umatrix_out")
  var urow_out_ptr mlpackArma
  urow_out := urow_out_ptr.ArmaToGonumUrow("urow_out")
  vector_out := GetParamVecInt("vector_out")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return col_out, double_out, int_out, matrix_and_info_out, matrix_out, model_bw_out, model_out, row_out, str_vector_out, string_out, ucol_out, umatrix_out, urow_out, vector_out
}
