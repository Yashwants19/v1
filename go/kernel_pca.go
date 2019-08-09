package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_kernel_pca
#include <capi/kernel_pca.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type Kernel_pcaOptionalParam struct {
    Bandwidth float64
    Center bool
    Copy_all_inputs bool
    Degree float64
    Kernel_scale float64
    New_dimensionality int
    Nystroem_method bool
    Offset float64
    Sampling string
    Verbose bool
}

func InitializeKernel_pca() *Kernel_pcaOptionalParam {
  return &Kernel_pcaOptionalParam{
    Bandwidth: 1,
    Center: false,
    Copy_all_inputs: false,
    Degree: 1,
    Kernel_scale: 1,
    New_dimensionality: 0,
    Nystroem_method: false,
    Offset: 0,
    Sampling: "kmeans",
    Verbose: false,
  }
}

/*
  This program performs Kernel Principal Components Analysis (KPCA) on the
  specified dataset with the specified kernel.  This will transform the data
  onto the kernel principal components, and optionally reduce the dimensionality
  by ignoring the kernel principal components with the smallest eigenvalues.
  
  For the case where a linear kernel is used, this reduces to regular PCA.
  
  For example, the following command will perform KPCA on the dataset input
  using the Gaussian kernel, and saving the transformed data to transformed: 
  
  param := InitializeKernel_pca()
  transformed := Kernel_pca(input, "gaussian", )
  
  The kernels that are supported are listed below:
  
   * 'linear': the standard linear dot product (same as normal PCA):
      K(x, y) = x^T y
  
   * 'gaussian': a Gaussian kernel; requires bandwidth:
      K(x, y) = exp(-(|| x - y || ^ 2) / (2 * (bandwidth ^ 2)))
  
   * 'polynomial': polynomial kernel; requires offset and degree:
      K(x, y) = (x^T y + offset) ^ degree
  
   * 'hyptan': hyperbolic tangent kernel; requires scale and offset:
      K(x, y) = tanh(scale * (x^T y) + offset)
  
   * 'laplacian': Laplacian kernel; requires bandwidth:
      K(x, y) = exp(-(|| x - y ||) / bandwidth)
  
   * 'epanechnikov': Epanechnikov kernel; requires bandwidth:
      K(x, y) = max(0, 1 - || x - y ||^2 / bandwidth^2)
  
   * 'cosine': cosine distance:
      K(x, y) = 1 - (x^T y) / (|| x || * || y ||)
  
  The parameters for each of the kernels should be specified with the options
  'bandwidth', 'kernel_scale', 'offset', or 'degree' (or a combination of those
  parameters).
  
  Optionally, the Nyström method ("Using the Nystroem method to speed up kernel
  machines", 2001) can be used to calculate the kernel matrix by specifying the
  'nystroem_method' parameter. This approach works by using a subset of the data
  as basis to reconstruct the kernel matrix; to specify the sampling scheme, the
  'sampling' parameter is used.  The sampling scheme for the Nyström method can
  be chosen from the following list: 'kmeans', 'random', 'ordered'.


  Input parameters:

   - input (mat.Dense): Input dataset to perform KPCA on.
   - kernel (string): The kernel to use; see the above documentation for
        the list of usable kernels.
   - bandwidth (float64): Bandwidth, for 'gaussian' and 'laplacian'
        kernels.  Default value 1.
   - center (bool): If set, the transformed data will be centered about
        the origin.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - degree (float64): Degree of polynomial, for 'polynomial' kernel. 
        Default value 1.
   - kernel_scale (float64): Scale, for 'hyptan' kernel.  Default value
        1.
   - new_dimensionality (int): If not 0, reduce the dimensionality of the
        output dataset by ignoring the dimensions with the smallest eigenvalues.
         Default value 0.
   - nystroem_method (bool): If set, the Nystroem method will be used.
   - offset (float64): Offset, for 'hyptan' and 'polynomial' kernels. 
        Default value 0.
   - sampling (string): Sampling scheme to use for the Nystroem method:
        'kmeans', 'random', 'ordered'  Default value 'kmeans'.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Matrix to save modified dataset to.

*/
func Kernel_pca(input *mat.Dense, kernel string, param *Kernel_pcaOptionalParam) (*mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Kernel Principal Components Analysis")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  GonumToArmaMat("input", input)
  SetPassed("input")

  // Detect if the parameter was passed; set if so.
  SetParamString("kernel", kernel)
  SetPassed("kernel")

  // Detect if the parameter was passed; set if so.
  if param.Bandwidth != 1 {
    SetParamDouble("bandwidth", param.Bandwidth)
    SetPassed("bandwidth")
  }

  // Detect if the parameter was passed; set if so.
  if param.Center != false {
    SetParamBool("center", param.Center)
    SetPassed("center")
  }

  // Detect if the parameter was passed; set if so.
  if param.Degree != 1 {
    SetParamDouble("degree", param.Degree)
    SetPassed("degree")
  }

  // Detect if the parameter was passed; set if so.
  if param.Kernel_scale != 1 {
    SetParamDouble("kernel_scale", param.Kernel_scale)
    SetPassed("kernel_scale")
  }

  // Detect if the parameter was passed; set if so.
  if param.New_dimensionality != 0 {
    SetParamInt("new_dimensionality", param.New_dimensionality)
    SetPassed("new_dimensionality")
  }

  // Detect if the parameter was passed; set if so.
  if param.Nystroem_method != false {
    SetParamBool("nystroem_method", param.Nystroem_method)
    SetPassed("nystroem_method")
  }

  // Detect if the parameter was passed; set if so.
  if param.Offset != 0 {
    SetParamDouble("offset", param.Offset)
    SetPassed("offset")
  }

  // Detect if the parameter was passed; set if so.
  if param.Sampling != "kmeans" {
    SetParamString("sampling", param.Sampling)
    SetPassed("sampling")
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
  C.mlpackkernel_pca()

  // Initialize result variable and get output.
  var output_ptr mlpackArma
  output := output_ptr.ArmaToGonumMat("output")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output
}
