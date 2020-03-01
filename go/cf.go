package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_cf
#include <capi/cf.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "unsafe" 
)

type CfOptionalParam struct {
    Algorithm string
    AllUserRecommendations bool
    InputModel *cfModel
    Interpolation string
    IterationOnlyTermination bool
    MaxIterations int
    MinResidue float64
    NeighborSearch string
    Neighborhood int
    Normalization string
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
    AllUserRecommendations: false,
    InputModel: nil,
    Interpolation: "average",
    IterationOnlyTermination: false,
    MaxIterations: 1000,
    MinResidue: 1e-05,
    NeighborSearch: "euclidean",
    Neighborhood: 5,
    Normalization: "none",
    Query: nil,
    Rank: 0,
    Recommendations: 5,
    Seed: 0,
    Test: nil,
    Training: nil,
    Verbose: false,
  }
}

type cfModel struct {
  mem unsafe.Pointer
}

func (m *cfModel) allocCFModel(identifier string) {
  m.mem = C.mlpackGetCFModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *cfModel) getCFModel(identifier string) {
  m.allocCFModel(identifier)
}

func setCFModel(identifier string, ptr *cfModel) {
  C.mlpackSetCFModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program performs collaborative filtering (CF) on the given dataset. Given
  a list of user, item and preferences (the "training" parameter), the program
  will perform a matrix decomposition and then can perform a series of actions
  related to collaborative filtering.  Alternately, the program can load an
  existing saved CF model with the "input_model" parameter and then use that
  model to provide recommendations or predict values.
  
  The input matrix should be a 3-dimensional matrix of ratings, where the first
  dimension is the user, the second dimension is the item, and the third
  dimension is that user's rating of that item.  Both the users and items should
  be numeric indices, not names. The indices are assumed to start from 0.
  
  A set of query users for which recommendations can be generated may be
  specified with the "query" parameter; alternately, recommendations may be
  generated for every user in the dataset by specifying the
  "all_user_recommendations" parameter.  In addition, the number of
  recommendations per user to generate can be specified with the
  "recommendations" parameter, and the number of similar users (the size of the
  neighborhood) to be considered when generating recommendations can be
  specified with the "neighborhood" parameter.
  
  For performing the matrix decomposition, the following optimization algorithms
  can be specified via the "algorithm" parameter: 
   - 'RegSVD' -- Regularized SVD using a SGD optimizer
   - 'NMF' -- Non-negative matrix factorization with alternating least squares
  update rules
   - 'BatchSVD' -- SVD batch learning
   - 'SVDIncompleteIncremental' -- SVD incomplete incremental learning
   - 'SVDCompleteIncremental' -- SVD complete incremental learning
   - 'BiasSVD' -- Bias SVD using a SGD optimizer
   - 'SVDPP' -- SVD++ using a SGD optimizer
  
  
  The following neighbor search algorithms can be specified via the
  "neighbor_search" parameter:
   - 'cosine'  -- Cosine Search Algorithm
   - 'euclidean'  -- Euclidean Search Algorithm
   - 'pearson'  -- Pearson Search Algorithm
  
  
  The following weight interpolation algorithms can be specified via the
  "interpolation" parameter:
   - 'average'  -- Average Interpolation Algorithm
   - 'regression'  -- Regression Interpolation Algorithm
   - 'similarity'  -- Similarity Interpolation Algorithm
  
  
  The following ranking normalization algorithms can be specified via the
  "normalization" parameter:
   - 'none'  -- No Normalization
   - 'item_mean'  -- Item Mean Normalization
   - 'overall_mean'  -- Overall Mean Normalization
   - 'user_mean'  -- User Mean Normalization
   - 'z_score'  -- Z-Score Normalization
  
  A trained model may be saved to with the "output_model" output parameter.
  
  To train a CF model on a dataset training_set using NMF for decomposition and
  saving the trained model to model, one could call: 
  
    param := mlpack.InitializeCf()
    param.Training = training_set
    param.Algorithm = "NMF"
    _, Model := mlpack.Cf(param)
  
  Then, to use this model to generate recommendations for the list of users in
  the query set users, storing 5 recommendations in recommendations, one could
  call 
  
    param := mlpack.InitializeCf()
    param.InputModel = &Model
    param.Query = users
    param.Recommendations = 5
    Recommendations, _ := mlpack.Cf(param)


  Input parameters:

   - Algorithm (string): Algorithm used for matrix factorization.  Default
        value 'NMF'.
   - AllUserRecommendations (bool): Generate recommendations for all
        users.
   - InputModel (cfModel): Trained CF model to load.
   - Interpolation (string): Algorithm used for weight interpolation. 
        Default value 'average'.
   - IterationOnlyTermination (bool): Terminate only when the maximum
        number of iterations is reached.
   - MaxIterations (int): Maximum number of iterations. If set to zero,
        there is no limit on the number of iterations.  Default value 1000.
   - MinResidue (float64): Residue required to terminate the factorization
        (lower values generally mean better fits).  Default value 1e-05.
   - NeighborSearch (string): Algorithm used for neighbor search.  Default
        value 'euclidean'.
   - Neighborhood (int): Size of the neighborhood of similar users to
        consider for each query user.  Default value 5.
   - Normalization (string): Normalization performed on the ratings. 
        Default value 'none'.
   - Query (mat.Dense): List of query users for which recommendations
        should be generated.
   - Rank (int): Rank of decomposed matrices (if 0, a heuristic is used to
        estimate the rank).  Default value 0.
   - Recommendations (int): Number of recommendations to generate for each
        query user.  Default value 5.
   - Seed (int): Set the random seed (0 uses std::time(NULL)).  Default
        value 0.
   - Test (mat.Dense): Test set to calculate RMSE on.
   - Training (mat.Dense): Input dataset to perform CF on.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - Output (mat.Dense): Matrix that will store output recommendations.
   - OutputModel (cfModel): Output for trained CF model.

 */
func Cf(param *CfOptionalParam) (*mat.Dense, cfModel) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Collaborative Filtering")

  // Detect if the parameter was passed; set if so.
  if param.Algorithm != "NMF" {
    setParamString("algorithm", param.Algorithm)
    setPassed("algorithm")
  }

  // Detect if the parameter was passed; set if so.
  if param.AllUserRecommendations != false {
    setParamBool("all_user_recommendations", param.AllUserRecommendations)
    setPassed("all_user_recommendations")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setCFModel("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Interpolation != "average" {
    setParamString("interpolation", param.Interpolation)
    setPassed("interpolation")
  }

  // Detect if the parameter was passed; set if so.
  if param.IterationOnlyTermination != false {
    setParamBool("iteration_only_termination", param.IterationOnlyTermination)
    setPassed("iteration_only_termination")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 1000 {
    setParamInt("max_iterations", param.MaxIterations)
    setPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.MinResidue != 1e-05 {
    setParamDouble("min_residue", param.MinResidue)
    setPassed("min_residue")
  }

  // Detect if the parameter was passed; set if so.
  if param.NeighborSearch != "euclidean" {
    setParamString("neighbor_search", param.NeighborSearch)
    setPassed("neighbor_search")
  }

  // Detect if the parameter was passed; set if so.
  if param.Neighborhood != 5 {
    setParamInt("neighborhood", param.Neighborhood)
    setPassed("neighborhood")
  }

  // Detect if the parameter was passed; set if so.
  if param.Normalization != "none" {
    setParamString("normalization", param.Normalization)
    setPassed("normalization")
  }

  // Detect if the parameter was passed; set if so.
  if param.Query != nil {
    gonumToArmaUmat("query", param.Query)
    setPassed("query")
  }

  // Detect if the parameter was passed; set if so.
  if param.Rank != 0 {
    setParamInt("rank", param.Rank)
    setPassed("rank")
  }

  // Detect if the parameter was passed; set if so.
  if param.Recommendations != 5 {
    setParamInt("recommendations", param.Recommendations)
    setPassed("recommendations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt("seed", param.Seed)
    setPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMat("test", param.Test)
    setPassed("test")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training != nil {
    gonumToArmaMat("training", param.Training)
    setPassed("training")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("output")
  setPassed("output_model")

  // Call the mlpack program.
  C.mlpackCf()

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  Output := outputPtr.armaToGonumUmat("output")
  var OutputModel cfModel
  OutputModel.getCFModel("output_model")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return Output, OutputModel
}
