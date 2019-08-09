package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lm -lmlpack -lmlpack_go_preprocess_describe
#include <capi/preprocess_describe.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type Preprocess_describeOptionalParam struct {
    Copy_all_inputs bool
    Dimension int
    Population bool
    Precision int
    Row_major bool
    Verbose bool
    Width int
}

func InitializePreprocess_describe() *Preprocess_describeOptionalParam {
  return &Preprocess_describeOptionalParam{
    Copy_all_inputs: false,
    Dimension: 0,
    Population: false,
    Precision: 4,
    Row_major: false,
    Verbose: false,
    Width: 8,
  }
}

/*
  This utility takes a dataset and prints out the descriptive statistics of the
  data. Descriptive statistics is the discipline of quantitatively describing
  the main features of a collection of information, or the quantitative
  description itself. The program does not modify the original file, but instead
  prints out the statistics to the console. The printed result will look like a
  table.
  
  Optionally, width and precision of the output can be adjusted by a user using
  the 'width' and 'precision' parameters. A user can also select a specific
  dimension to analyze if there are too many dimensions. The 'population'
  parameter can be specified when the dataset should be considered as a
  population.  Otherwise, the dataset will be considered as a sample.
  
  So, a simple example where we want to print out statistical facts about the
  dataset X using the default settings, we could run 
  
  param := InitializePreprocess_describe()
  param.Verbose = true
   := Preprocess_describe(X, param)
  
  If we want to customize the width to 10 and precision to 5 and consider the
  dataset as a population, we could run
  
  param := InitializePreprocess_describe()
  param.Width = 10
  param.Precision = 5
  param.Verbose = true
   := Preprocess_describe(X, param)


  Input parameters:

   - input (mat.Dense): Matrix containing data,
   - copy_all_inputs (bool): If specified, all input parameters will be
        deep copied before the method is run.  This is useful for debugging
        problems where the input parameters are being modified by the algorithm,
        but can slow down the code.
   - dimension (int): Dimension of the data. Use this to specify a
        dimension  Default value 0.
   - population (bool): If specified, the program will calculate
        statistics assuming the dataset is the population. By default, the
        program will assume the dataset as a sample.
   - precision (int): Precision of the output statistics.  Default value
        4.
   - row_major (bool): If specified, the program will calculate statistics
        across rows, not across columns.  (Remember that in mlpack, a column
        represents a point, so this option is generally not necessary.)
   - verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.
   - width (int): Width of the output table.  Default value 8.

  Output parameters:


*/
func Preprocess_describe(input *mat.Dense, param *Preprocess_describeOptionalParam) () {
  ResetTimers()
  EnableTimers()
  DisableBacktrace()
  DisableVerbose()
  RestoreSettings("Descriptive Statistics")

  // Detect if the parameter was passed; set if so.
  if param.Copy_all_inputs == true {
    SetParamBool("copy_all_inputs", param.Copy_all_inputs)
    SetPassed("copy_all_inputs")
  }

  // Detect if the parameter was passed; set if so.
  GonumToArmaMat("input", input)
  SetPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Dimension != 0 {
    SetParamInt("dimension", param.Dimension)
    SetPassed("dimension")
  }

  // Detect if the parameter was passed; set if so.
  if param.Population != false {
    SetParamBool("population", param.Population)
    SetPassed("population")
  }

  // Detect if the parameter was passed; set if so.
  if param.Precision != 4 {
    SetParamInt("precision", param.Precision)
    SetPassed("precision")
  }

  // Detect if the parameter was passed; set if so.
  if param.Row_major != false {
    SetParamBool("row_major", param.Row_major)
    SetPassed("row_major")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    SetParamBool("verbose", param.Verbose)
    SetPassed("verbose")
    EnableVerbose()
  }

  // Detect if the parameter was passed; set if so.
  if param.Width != 8 {
    SetParamInt("width", param.Width)
    SetPassed("width")
  }

  // Mark all output options as passed.

  // Call the mlpack program.
  C.mlpackpreprocess_describe()

  // Initialize result variable and get output.

  // Clear settings.
  ClearSettings()

  // Return output(s).
  return 
}
