package main

import (
  "github.com/frictionlessdata/tableschema-go/csv"
  "github.com/Yashwants19/v1/mlpack"
  "gonum.org/v1/gonum/mat"
  "fmt"
  "os"
)
func main() {

  // Download dataset.
  mlpack.DownloadFile("https://www.mlpack.org/datasets/ml-20m/ratings-only.csv.gz",
                      "ratings-only.csv.gz")
  mlpack.DownloadFile("https://www.mlpack.org/datasets/ml-20m/movies.csv.gz",
                      "movies.csv.gz")

  // Extract dataset.
  mlpack.UnZip("ratings-only.csv.gz", "ratings-only.csv")
  f1, _ := os.Open("ratings-only.csv")
  defer f1.Close()
  data := mlpack.NewReader(f1)
  _, _ = data.ReadHeading()
  ratings, _ := data.ReadAll()

  //mlpack.UnZip("movies.csv.gz", "movies.csv")
  table, _ := csv.NewTable(csv.FromFile("movies.csv"), csv.LoadHeaders())
  movies, _ := table.ReadColumn("title")

  fmt.Println(ratings.Dims())
  // Split the dataset using mlpack.
  params := mlpack.PreprocessSplitOptions()
  params.TestRatio = 0.1
  params.Verbose = true
  _, _, ratings_train, _ :=
      mlpack.PreprocessSplit(ratings, params)
  // Train the model.  Change the rank to increase/decrease the complexity of the
  // model.
  cf_params := mlpack.CfOptions()
  cf_params.Training = ratings_train
  //cf_params.Test = ratings_test
  cf_params.MaxIterations = 5
  cf_params.Rank = 10
  cf_params.Verbose = true
  cf_params.Algorithm = "RegSVD"
  _, cf_model := mlpack.Cf(cf_params)

  // Now query the 5 top movies for user 1.
  cf_params_2 := mlpack.CfOptions()
  cf_params_2.InputModel = &cf_model
  cf_params_2.Recommendations = 10
  cf_params_2.Query = mat.NewDense(1, 1, []float64{1})
  cf_params_2.Verbose = true
  cf_params_2.MaxIterations = 5
  output, _ := mlpack.Cf(cf_params_2)

  // Get the names of the movies for user 1.
  fmt.Print("Recommendations for user 1")
  fmt.Println()
  for i := 0; i < 10; i++ {
    fmt.Println(i, ":", movies[int(output.At(i , 0))])
  }
}
