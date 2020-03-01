package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_preprocess_split
#include <capi/preprocess_split.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type PreprocessSplitOptionalParam struct {
    InputLabels *mat.Dense
    Seed int
    TestRatio float64
    Verbose bool
}

func InitializePreprocessSplit() *PreprocessSplitOptionalParam {
  return &PreprocessSplitOptionalParam{
    InputLabels: nil,
    Seed: 0,
    TestRatio: 0.2,
    Verbose: false,
  }
}

/*
  This utility takes a dataset and optionally labels and splits them into a
  training set and a test set. Before the split, the points in the dataset are
  randomly reordered. The percentage of the dataset to be used as the test set
  can be specified with the "test_ratio" parameter; the default is 0.2 (20%).
  
  The output training and test matrices may be saved with the "training" and
  "test" output parameters.
  
  Optionally, labels can be also be split along with the data by specifying the
  "input_labels" parameter.  Splitting labels works the same way as splitting
  the data. The output training and test labels may be saved with the
  "training_labels" and "test_labels" output parameters, respectively.
  
  So, a simple example where we want to split the dataset X into X_train and
  X_test with 60% of the data in the training set and 40% of the dataset in the
  test set, we could run 
  
    param := mlpack.InitializePreprocessSplit()
    param.TestRatio = 0.4
    XTest, _, XTrain, _ := mlpack.PreprocessSplit(X, param)
  
  If we had a dataset X and associated labels y, and we wanted to split these
  into X_train, y_train, X_test, and y_test, with 30% of the data in the test
  set, we could run
  
    param := mlpack.InitializePreprocessSplit()
    param.InputLabels = y
    param.TestRatio = 0.3
    XTest, YTest, XTrain, YTrain := mlpack.PreprocessSplit(X, param)


  Input parameters:

   - Input (mat.Dense): Matrix containing data.
   - InputLabels (mat.Dense): Matrix containing labels.
   - Seed (int): Random seed (0 for std::time(NULL)).  Default value 0.
   - TestRatio (float64): Ratio of test set; if not set,the ratio defaults
        to 0.2  Default value 0.2.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - Test (mat.Dense): Matrix to save test data to.
   - TestLabels (mat.Dense): Matrix to save test labels to.
   - Training (mat.Dense): Matrix to save training data to.
   - TrainingLabels (mat.Dense): Matrix to save train labels to.

 */
func PreprocessSplit(input *mat.Dense, param *PreprocessSplitOptionalParam) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Split Data")

  // Detect if the parameter was passed; set if so.
  gonumToArmaMat("input", input)
  setPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.InputLabels != nil {
    gonumToArmaUmat("input_labels", param.InputLabels)
    setPassed("input_labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt("seed", param.Seed)
    setPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.TestRatio != 0.2 {
    setParamDouble("test_ratio", param.TestRatio)
    setPassed("test_ratio")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("test")
  setPassed("test_labels")
  setPassed("training")
  setPassed("training_labels")

  // Call the mlpack program.
  C.mlpackPreprocessSplit()

  // Initialize result variable and get output.
  var testPtr mlpackArma
  Test := testPtr.armaToGonumMat("test")
  var testLabelsPtr mlpackArma
  TestLabels := testLabelsPtr.armaToGonumUmat("test_labels")
  var trainingPtr mlpackArma
  Training := trainingPtr.armaToGonumMat("training")
  var trainingLabelsPtr mlpackArma
  TrainingLabels := trainingLabelsPtr.armaToGonumUmat("training_labels")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return Test, TestLabels, Training, TrainingLabels
}
