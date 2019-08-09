package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_kmeans
#include <capi/kmeans.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type KmeansOptionalParam struct {
    Algorithm string
    Allow_empty_clusters bool
    Copy_all_inputs bool
    In_place bool
    Initial_centroids *mat.Dense
    Kill_empty_clusters bool
    Labels_only bool
    Max_iterations int
    Percentage float64
    Refined_start bool
    Samplings int
    Seed int
    Verbose bool
}

func InitializeKmeans() *KmeansOptionalParam {
  return &KmeansOptionalParam{
    Algorithm: "naive",
    Allow_empty_clusters: false,
    Copy_all_inputs: false,
    In_place: false,
    Initial_centroids: nil,
    Kill_empty_clusters: false,
    Labels_only: false,
    Max_iterations: 1000,
    Percentage: 0.02,
    Refined_start: false,
    Samplings: 100,
    Seed: 0,
    Verbose: false,
  }
}

/*
  This program performs K-Means clustering on the given dataset.  It can return
  the learned cluster assignments, and the centroids of the clusters.  Empty
  clusters are not allowed by default; when a cluster becomes empty, the point
  furthest from the centroid of the cluster with maximum variance is taken to
  fill that cluster.
  
  Optionally, the Bradley and Fayyad approach ("Refining initial points for
  k-means clustering", 1998) can be used to select initial points by specifying
  the 'refined_start' parameter.  This approach works by taking random samplings
  of the dataset; to specify the number of samplings, the 'samplings' parameter
  is used, and to specify the percentage of the dataset to be used in each
  sample, the 'percentage' parameter is used (it should be a value between 0.0
  and 1.0).
  
  There are several options available for the algorithm used for each Lloyd
  iteration, specified with the 'algorithm'  option.  The standard O(kN)
  approach can be used ('naive').  Other options include the Pelleg-Moore
  tree-based algorithm ('pelleg-moore'), Elkan's triangle-inequality based
  algorithm ('elkan'), Hamerly's modification to Elkan's algorithm ('hamerly'),
  the dual-tree k-means algorithm ('dualtree'), and the dual-tree k-means
  algorithm using the cover tree ('dualtree-covertree').
  
  The behavior for when an empty cluster is encountered can be modified with the
  'allow_empty_clusters' option.  When this option is specified and there is a
  cluster owning no points at the end of an iteration, that cluster's centroid
  will simply remain in its position from the previous iteration. If the
  'kill_empty_clusters' option is specified, then when a cluster owns no points
  at the end of an iteration, the cluster centroid is simply filled with
  DBL_MAX, killing it and effectively reducing k for the rest of the
  computation.  Note that the default option when neither empty cluster option
  is specified can be time-consuming to calculate; therefore, specifying either
  of these parameters will often accelerate runtime.
  
  Initial clustering assignments may be specified using the 'initial_centroids'
  parameter, and the maximum number of iterations may be specified with the
  'max_iterations' parameter.
  
  As an example, to use Hamerly's algorithm to perform k-means clustering with
  k=10 on the dataset data, saving the centroids to centroids and the
  assignments for each point to assignments, the following command could be
  used:
  
  param := InitializeKmeans()
  centroids, assignments := Kmeans(data, 10, )
  
  To run k-means on that same dataset with initial centroids specified in
  initial with a maximum of 500 iterations, storing the output centroids in
  final the following command may be used:
  
  param := InitializeKmeans()
  param.Initial_centroids = initial
  param.Max_iterations = 500
  final, _ := Kmeans(data, 10, param)


  Input parameters:

   - clusters (int): Number of clusters to find (0 autodetects from
        initial centroids).
   - input (mat.Dense): Input dataset to perform clustering on.
   - algorithm (string): Algorithm to use for the Lloyd iteration
        ('naive', 'pelleg-moore', 'elkan', 'hamerly', 'dualtree', or
        'dualtree-covertree').  Default value 'naive'.
   - allow_empty_clusters (bool): Allow empty clusters to be persist.
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - in_place (bool): If specified, a column containing the learned
        cluster assignments will be added to the input dataset file.  In this
        case, --output_file is overridden. (Do not use in Python.)
   - initial_centroids (mat.Dense): Start with the specified initial
        centroids.
   - kill_empty_clusters (bool): Remove empty clusters when they occur.
   - labels_only (bool): Only output labels into output file.
   - max_iterations (int): Maximum number of iterations before k-means
        terminates.  Default value 1000.
   - percentage (float64): Percentage of dataset to use for each refined
        start sampling (use when --refined_start is specified).  Default value
        0.02.
   - refined_start (bool): Use the refined initial point strategy by
        Bradley and Fayyad to choose initial points.
   - samplings (int): Number of samplings to perform for refined start
        (use when --refined_start is specified).  Default value 100.
   - seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - centroid (mat.Dense): If specified, the centroids of each cluster
        will  be written to the given file.
   - output (mat.Dense): Matrix to store output labels or labeled data
        to.

*/
func Kmeans(clusters int, input *mat.Dense, param *KmeansOptionalParam) (*mat.Dense, *mat.Dense) {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("K-Means Clustering")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  SetParamInt("clusters", clusters)
  SetPassed("clusters")

  // Detect if the parameter was passed; set if so.
  GonumToArmaMat("input", input)
  SetPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Algorithm != "naive" {
    SetParamString("algorithm", param.Algorithm)
    SetPassed("algorithm")
  }

  // Detect if the parameter was passed; set if so.
  if param.Allow_empty_clusters != false {
    SetParamBool("allow_empty_clusters", param.Allow_empty_clusters)
    SetPassed("allow_empty_clusters")
  }

  // Detect if the parameter was passed; set if so.
  if param.In_place != false {
    SetParamBool("in_place", param.In_place)
    SetPassed("in_place")
  }

  // Detect if the parameter was passed; set if so.
  if param.Initial_centroids != nil {
    GonumToArmaMat("initial_centroids", param.Initial_centroids)
    SetPassed("initial_centroids")
  }

  // Detect if the parameter was passed; set if so.
  if param.Kill_empty_clusters != false {
    SetParamBool("kill_empty_clusters", param.Kill_empty_clusters)
    SetPassed("kill_empty_clusters")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels_only != false {
    SetParamBool("labels_only", param.Labels_only)
    SetPassed("labels_only")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max_iterations != 1000 {
    SetParamInt("max_iterations", param.Max_iterations)
    SetPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Percentage != 0.02 {
    SetParamDouble("percentage", param.Percentage)
    SetPassed("percentage")
  }

  // Detect if the parameter was passed; set if so.
  if param.Refined_start != false {
    SetParamBool("refined_start", param.Refined_start)
    SetPassed("refined_start")
  }

  // Detect if the parameter was passed; set if so.
  if param.Samplings != 100 {
    SetParamInt("samplings", param.Samplings)
    SetPassed("samplings")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    SetParamInt("seed", param.Seed)
    SetPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Mark all output options as passed.
  SetPassed("centroid")
  SetPassed("output")

  // Call the mlpack program.
  C.mlpackkmeans()

  // Initialize result variable and get output.
  var centroid_ptr mlpackArma
  centroid := centroid_ptr.ArmaToGonumMat("centroid")
  var output_ptr mlpackArma
  output := output_ptr.ArmaToGonumMat("output")

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return centroid, output
}
