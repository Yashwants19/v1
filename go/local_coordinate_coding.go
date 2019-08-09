package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_local_coordinate_coding
#include <capi/local_coordinate_coding.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type Local_coordinate_codingOptionalParam struct {
    Atoms int
    Copy_all_inputs bool
    Initial_dictionary *mat.Dense
    Input_model *LocalCoordinateCoding
    Lambda float64
    Max_iterations int
    Normalize bool
    Seed int
    Test *mat.Dense
    Tolerance float64
    Training *mat.Dense
    Verbose bool
}

func InitializeLocal_coordinate_coding() *Local_coordinate_codingOptionalParam {
  return &Local_coordinate_codingOptionalParam{
    Atoms: 0,
    Copy_all_inputs: false,
    Initial_dictionary: nil,
    Input_model: nil,
    Lambda: 0,
    Max_iterations: 0,
    Normalize: false,
    Seed: 0,
    Test: nil,
    Tolerance: 0.01,
    Training: nil,
    Verbose: false,
  }
}

type LocalCoordinateCoding struct {
 mem unsafe.Pointer
}

func (m *LocalCoordinateCoding) allocLocalCoordinateCoding(identifier string) {
 m.mem = C.mlpackGetLocalCoordinateCodingPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *LocalCoordinateCoding) getLocalCoordinateCoding(identifier string) {
 m.allocLocalCoordinateCoding(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setLocalCoordinateCoding(identifier string, ptr *LocalCoordinateCoding) {
 C.mlpackSetLocalCoordinateCodingPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  An implementation of Local Coordinate Coding (LCC), which codes data that
  approximately lives on a manifold using a variation of l1-norm regularized
  sparse coding.  Given a dense data matrix X with n points and d dimensions,
  LCC seeks to find a dense dictionary matrix D with k atoms in d dimensions,
  and a coding matrix Z with n points in k dimensions.  Because of the
  regularization method used, the atoms in D should lie close to the manifold on
  which the data points lie.
  
  The original data matrix X can then be reconstructed as D * Z.  Therefore,
  this program finds a representation of each point in X as a sparse linear
  combination of atoms in the dictionary D.
  
  The coding is found with an algorithm which alternates between a dictionary
  step, which updates the dictionary D, and a coding step, which updates the
  coding matrix Z.
  
  To run this program, the input matrix X must be specified (with -i), along
  with the number of atoms in the dictionary (-k).  An initial dictionary may
  also be specified with the 'initial_dictionary' parameter.  The l1-norm
  regularization parameter is specified with the 'lambda' parameter.  For
  example, to run LCC on the dataset data using 200 atoms and an
  l1-regularization parameter of 0.1, saving the dictionary 'dictionary' and the
  codes into 'codes', use
  
  param := InitializeLocal_coordinate_coding()
  param.Training = data
  param.Atoms = 200
  param.Lambda = 0.1
  codes, dict, _ := Local_coordinate_coding(param)
  
  The maximum number of iterations may be specified with the 'max_iterations'
  parameter. Optionally, the input data matrix X can be normalized before coding
  with the 'normalize' parameter.
  
  An LCC model may be saved using the 'output_model' output parameter.  Then, to
  encode new points from the dataset points with the previously saved model
  lcc_model, saving the new codes to new_codes, the following command can be
  used:
  
  param := InitializeLocal_coordinate_coding()
  param.Input_model = lcc_model
  param.Test = points
  new_codes, _, _ := Local_coordinate_coding(param)


  Input parameters:

   - atoms (int): Number of atoms in the dictionary.  Default value 0.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - initial_dictionary (mat.Dense): Optional initial dictionary.
   - input_model (LocalCoordinateCoding): Input LCC model.
   - lambda (float64): Weighted l1-norm regularization parameter.  Default
        value 0.
   - max_iterations (int): Maximum number of iterations for LCC (0
        indicates no limit).  Default value 0.
   - normalize (bool): If set, the input data matrix will be normalized
        before coding.
   - seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - test (mat.Dense): Test points to encode.
   - tolerance (float64): Tolerance for objective function.  Default value
        0.01.
   - training (mat.Dense): Matrix of training data (X).
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - codes (mat.Dense): Output codes matrix.
   - dictionary (mat.Dense): Output dictionary matrix.
   - output_model (LocalCoordinateCoding): Output for trained LCC model.

*/
func Local_coordinate_coding(param *Local_coordinate_codingOptionalParam) (*mat.Dense, *mat.Dense, LocalCoordinateCoding) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Local Coordinate Coding")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Atoms != 0 {
    SetParamInt("atoms", param.Atoms)
    SetPassed("atoms")
  }

  // Detect if the parameter was passed; set if so.
  if param.Initial_dictionary != nil {
    GonumToArmaMat("initial_dictionary", param.Initial_dictionary)
    SetPassed("initial_dictionary")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setLocalCoordinateCoding("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda != 0 {
    SetParamDouble("lambda", param.Lambda)
    SetPassed("lambda")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max_iterations != 0 {
    SetParamInt("max_iterations", param.Max_iterations)
    SetPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Normalize != false {
    SetParamBool("normalize", param.Normalize)
    SetPassed("normalize")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    SetParamInt("seed", param.Seed)
    SetPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    GonumToArmaMat("test", param.Test)
    SetPassed("test")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tolerance != 0.01 {
    SetParamDouble("tolerance", param.Tolerance)
    SetPassed("tolerance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training != nil {
    GonumToArmaMat("training", param.Training)
    SetPassed("training")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("codes")
  SetPassed("dictionary")
  SetPassed("output_model")

  // Call the mlpack program.
  C.mlpacklocal_coordinate_coding()

  // Initialize result variable and get output.
  var codes_ptr mlpackArma
  codes := codes_ptr.ArmaToGonumMat("codes")
  var dictionary_ptr mlpackArma
  dictionary := dictionary_ptr.ArmaToGonumMat("dictionary")
  var output_model LocalCoordinateCoding
  output_model.getLocalCoordinateCoding("output_model")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return codes, dictionary, output_model
}
