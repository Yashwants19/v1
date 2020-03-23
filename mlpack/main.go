// What it does:
//
// 	This program outputs the current OpenCV library version to the console.
//
// How to run:
//
// 		go run main.go
//
// +build example

package main

import (
	"fmt"

	"github.com/Yashwants19/v1/mlpack"
)

func main() {
	fmt.Printf("mlpack version: %s\n", mlpack.Version())
}
