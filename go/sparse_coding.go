package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_sparse_coding
#include <capi/sparse_coding.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type Sparse_codingOptionalParam struct {
    Atoms int
    Copy_all_inputs bool
    Initial_dictionary *mat.Dense
    Input_model *SparseCoding
    Lambda1 float64
    Lambda2 float64
    Max_iterations int
    Newton_tolerance float64
    Normalize bool
    Objective_tolerance float64
    Seed int
    Test *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func InitializeSparse_coding() *Sparse_codingOptionalParam {
  return &Sparse_codingOptionalParam{
    Atoms: 15,
    Copy_all_inputs: false,
    Initial_dictionary: nil,
    Input_model: nil,
    Lambda1: 0,
    Lambda2: 0,
    Max_iterations: 0,
    Newton_tolerance: 1e-06,
    Normalize: false,
    Objective_tolerance: 0.01,
    Seed: 0,
    Test: nil,
    Training: nil,
    Verbose: false,
  }
}

type SparseCoding struct {
 mem unsafe.Pointer
}

func (m *SparseCoding) allocSparseCoding(identifier string) {
 m.mem = C.mlpackGetSparseCodingPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *SparseCoding) getSparseCoding(identifier string) {
 m.allocSparseCoding(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setSparseCoding(identifier string, ptr *SparseCoding) {
 C.mlpackSetSparseCodingPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  An implementation of Sparse Coding with Dictionary Learning, which achieves
  sparsity via an l1-norm regularizer on the codes (LASSO) or an (l1+l2)-norm
  regularizer on the codes (the Elastic Net).  Given a dense data matrix X with
  d dimensions and n points, sparse coding seeks to find a dense dictionary
  matrix D with k atoms in d dimensions, and a sparse coding matrix Z with n
  points in k dimensions.
  
  The original data matrix X can then be reconstructed as Z * D.  Therefore,
  this program finds a representation of each point in X as a sparse linear
  combination of atoms in the dictionary D.
  
  The sparse coding is found with an algorithm which alternates between a
  dictionary step, which updates the dictionary D, and a sparse coding step,
  which updates the sparse coding matrix.
  
  Once a dictionary D is found, the sparse coding model may be used to encode
  other matrices, and saved for future usage.
  
  To run this program, either an input matrix or an already-saved sparse coding
  model must be specified.  An input matrix may be specified with the 'training'
  option, along with the number of atoms in the dictionary (specified with the
  'atoms' parameter).  It is also possible to specify an initial dictionary for
  the optimization, with the 'initial_dictionary' parameter.  An input model may
  be specified with the 'input_model' parameter.
  
  As an example, to build a sparse coding model on the dataset data using 200
  atoms and an l1-regularization parameter of 0.1, saving the model into model,
  use 
  
  param := InitializeSparse_coding()
  param.Training = data
  param.Atoms = 200
  param.Lambda1 = 0.1
  _, _, model := Sparse_coding(param)
  
  Then, this model could be used to encode a new matrix, otherdata, and save the
  output codes to codes: 
  
  param := InitializeSparse_coding()
  param.Input_model = model
  param.Test = otherdata
  codes, _, _ := Sparse_coding(param)


  Input parameters:

   - atoms (int): Number of atoms in the dictionary.  Default value 15.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - initial_dictionary (mat.Dense): Optional initial dictionary matrix.
   - input_model (SparseCoding): File containing input sparse coding
        model.
   - lambda1 (float64): Sparse coding l1-norm regularization parameter. 
        Default value 0.
   - lambda2 (float64): Sparse coding l2-norm regularization parameter. 
        Default value 0.
   - max_iterations (int): Maximum number of iterations for sparse coding
        (0 indicates no limit).  Default value 0.
   - newton_tolerance (float64): Tolerance for convergence of Newton
        method.  Default value 1e-06.
   - normalize (bool): If set, the input data matrix will be normalized
        before coding.
   - objective_tolerance (float64): Tolerance for convergence of the
        objective function.  Default value 0.01.
   - seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - test (mat.Dense): Optional matrix to be encoded by trained model.
   - training (mat.Dense): Matrix of training data (X).
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - codes (mat.Dense): Matrix to save the output sparse codes of the test
        matrix (--test_file) to.
   - dictionary (mat.Dense): Matrix to save the output dictionary to.
   - output_model (SparseCoding): File to save trained sparse coding model
        to.

*/
func Sparse_coding(param *Sparse_codingOptionalParam) (*mat.Dense, *mat.Dense, SparseCoding) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Sparse Coding")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Atoms != 15 {
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
    setSparseCoding("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda1 != 0 {
    SetParamDouble("lambda1", param.Lambda1)
    SetPassed("lambda1")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda2 != 0 {
    SetParamDouble("lambda2", param.Lambda2)
    SetPassed("lambda2")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max_iterations != 0 {
    SetParamInt("max_iterations", param.Max_iterations)
    SetPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Newton_tolerance != 1e-06 {
    SetParamDouble("newton_tolerance", param.Newton_tolerance)
    SetPassed("newton_tolerance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Normalize != false {
    SetParamBool("normalize", param.Normalize)
    SetPassed("normalize")
  }

  // Detect if the parameter was passed; set if so.
  if param.Objective_tolerance != 0.01 {
    SetParamDouble("objective_tolerance", param.Objective_tolerance)
    SetPassed("objective_tolerance")
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
  C.mlpacksparse_coding()

  // Initialize result variable and get output.
  var codes_ptr mlpackArma
  codes := codes_ptr.ArmaToGonumMat("codes")
  var dictionary_ptr mlpackArma
  dictionary := dictionary_ptr.ArmaToGonumMat("dictionary")
  var output_model SparseCoding
  output_model.getSparseCoding("output_model")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return codes, dictionary, output_model
}
