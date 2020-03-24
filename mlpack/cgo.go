// +build !customenv

package mlpack

/*
#cgo CFLAGS: -I. -I/capi -g -Wall -Wno-unused-variable
#cgo !windows pkg-config: mlpack
*/
import "C"
