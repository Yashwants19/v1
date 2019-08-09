package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_cf
#include <capi/cf.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "time" 
  "unsafe" 
)

type CfOptionalParam struct {
    Algorithm string
    All_user_recommendations bool
    Copy_all_inputs bool
    Input_model *CFModel
    Interpolation string
    Iteration_only_termination bool
    Max_iterations int
    Min_residue float64
    Neighbor_search string
    Neighborhood int
    Query *mat.Dense
    Rank int
    Recommendations int
    Seed int
    Test *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func InitializeCf() *CfOptionalParam {
  return &CfOptionalParam{
    Algorithm: "NMF",
    All_user_recommendations: false,
    Copy_all_inputs: false,
    Input_model: nil,
    Interpolation: "average",
    Iteration_only_termination: false,
    Max_iterations: 1000,
    Min_residue: 1e-05,
    Neighbor_search: "euclidean",
    Neighborhood: 5,
    Query: nil,
    Rank: 0,
    Recommendations: 5,
    Seed: 0,
    Test: nil,
    Training: nil,
    Verbose: false,
  }
}

type CFModel struct {
 mem unsafe.Pointer
}

func (m *CFModel) allocCFModel(identifier string) {
 m.mem = C.mlpackGetCFModelPtr(C.CString(identifier))
 runtime.KeepAlive(m)
}

func (m *CFModel) getCFModel(identifier string) {
 m.allocCFModel(identifier)
 time.Sleep(time.Second)
 runtime.GC()
 time.Sleep(time.Second)
}

func setCFModel(identifier string, ptr *CFModel) {
 C.mlpackSetCFModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program performs collaborative filtering (CF) on the given dataset. Given
  a list of user, item and preferences (the 'training' parameter), the program
  will perform a matrix decomposition and then can perform a series of actions
  related to collaborative filtering.  Alternately, the program can load an
  existing saved CF model with the 'input_model' parameter and then use that
  model to provide recommendations or predict values.
  
  The input matrix should be a 3-dimensional matrix of ratings, where the first
  dimension is the user, the second dimension is the item, and the third
  dimension is that user's rating of that item.  Both the users and items should
  be numeric indices, not names. The indices are assumed to start from 0.
  
  A set of query users for which recommendations can be generated may be
  specified with the 'query' parameter; alternately, recommendations may be
  generated for every user in the dataset by specifying the
  'all_user_recommendations' parameter.  In addition, the number of
  recommendations per user to generate can be specified with the
  'recommendations' parameter, and the number of similar users (the size of the
  neighborhood) to be considered when generating recommendations can be
  specified with the 'neighborhood' parameter.
  
  For performing the matrix decomposition, the following optimization algorithms
  can be specified via the 'algorithm' parameter: 
   - 'RegSVD' -- Regularized SVD using a SGD optimizer
   - 'NMF' -- Non-negative matrix factorization with alternating least squares
  update rules
   - 'BatchSVD' -- SVD batch learning
   - 'SVDIncompleteIncremental' -- SVD incomplete incremental learning
   - 'SVDCompleteIncremental' -- SVD complete incremental learning
   - 'BiasSVD' -- Bias SVD using a SGD optimizer
   - 'SVDPP' -- SVD++ using a SGD optimizer
  
  
  The following neighbor search algorithms can be specified via the
  'neighbor_search' parameter:
   - 'cosine'  -- Cosine Search Algorithm
   - 'euclidean'  -- Euclidean Search Algorithm
   - 'pearson'  -- Pearson Search Algorithm
  
  
  The following weight interpolation algorithms can be specified via the
  'interpolation' parameter:
   - 'average'  -- Average Interpolation Algorithm
   - 'regression'  -- Regression Interpolation Algorithm
   - 'similarity'  -- Similarity Interpolation Algorithm
  
  A trained model may be saved to with the 'output_model' output parameter.
  
  To train a CF model on a dataset training_set using NMF for decomposition and
  saving the trained model to model, one could call: 
  
  param := InitializeCf()
  param.Training = training_set
  param.Algorithm = "NMF"
  _, model := Cf(param)
  
  Then, to use this model to generate recommendations for the list of users in
  the query set users, storing 5 recommendations in recommendations, one could
  call 
  
  param := InitializeCf()
  param.Input_model = model
  param.Query = users
  param.Recommendations = 5
  recommendations, _ := Cf(param)


  Input parameters:

   - algorithm (string): Algorithm used for matrix factorization.  Default
        value 'NMF'.
   - all_user_recommendations (bool): Generate recommendations for all
        users.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - input_model (CFModel): Trained CF model to load.
   - interpolation (string): Algorithm used for weight interpolation. 
        Default value 'average'.
   - iteration_only_termination (bool): Terminate only when the maximum
        number of iterations is reached.
   - max_iterations (int): Maximum number of iterations. If set to zero,
        there is no limit on the number of iterations.  Default value 1000.
   - min_residue (float64): Residue required to terminate the
        factorization (lower values generally mean better fits).  Default value
        1e-05.
   - neighbor_search (string): Algorithm used for neighbor search. 
        Default value 'euclidean'.
   - neighborhood (int): Size of the neighborhood of similar users to
        consider for each query user.  Default value 5.
   - query (mat.Dense): List of query users for which recommendations
        should be generated.
   - rank (int): Rank of decomposed matrices (if 0, a heuristic is used to
        estimate the rank).  Default value 0.
   - recommendations (int): Number of recommendations to generate for each
        query user.  Default value 5.
   - seed (int): Set the random seed (0 uses std::time(NULL)).  Default
        value 0.
   - test (mat.Dense): Test set to calculate RMSE on.
   - training (mat.Dense): Input dataset to perform CF on.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Matrix that will store output recommendations.
   - output_model (CFModel): Output for trained CF model.

*/
func Cf(param *CfOptionalParam) (*mat.Dense, CFModel) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Collaborative Filtering")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  if param.Algorithm != "NMF" {
    SetParamString("algorithm", param.Algorithm)
    SetPassed("algorithm")
  }

  // Detect if the parameter was passed; set if so.
  if param.All_user_recommendations != false {
    SetParamBool("all_user_recommendations", param.All_user_recommendations)
    SetPassed("all_user_recommendations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input_model != nil {
    setCFModel("input_model", param.Input_model)
    SetPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Interpolation != "average" {
    SetParamString("interpolation", param.Interpolation)
    SetPassed("interpolation")
  }

  // Detect if the parameter was passed; set if so.
  if param.Iteration_only_termination != false {
    SetParamBool("iteration_only_termination", param.Iteration_only_termination)
    SetPassed("iteration_only_termination")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max_iterations != 1000 {
    SetParamInt("max_iterations", param.Max_iterations)
    SetPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Min_residue != 1e-05 {
    SetParamDouble("min_residue", param.Min_residue)
    SetPassed("min_residue")
  }

  // Detect if the parameter was passed; set if so.
  if param.Neighbor_search != "euclidean" {
    SetParamString("neighbor_search", param.Neighbor_search)
    SetPassed("neighbor_search")
  }

  // Detect if the parameter was passed; set if so.
  if param.Neighborhood != 5 {
    SetParamInt("neighborhood", param.Neighborhood)
    SetPassed("neighborhood")
  }

  // Detect if the parameter was passed; set if so.
  if param.Query != nil {
    GonumToArmaUmat("query", param.Query)
    SetPassed("query")
  }

  // Detect if the parameter was passed; set if so.
  if param.Rank != 0 {
    SetParamInt("rank", param.Rank)
    SetPassed("rank")
  }

  // Detect if the parameter was passed; set if so.
  if param.Recommendations != 5 {
    SetParamInt("recommendations", param.Recommendations)
    SetPassed("recommendations")
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
  SetPassed("output")
  SetPassed("output_model")

  // Call the mlpack program.
  C.mlpackcf()

  // Initialize result variable and get output.
  var output_ptr mlpackArma
  output := output_ptr.ArmaToGonumUmat("output")
  var output_model CFModel
  output_model.getCFModel("output_model")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return output, output_model
}
