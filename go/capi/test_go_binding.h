#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetGaussianKernelPtr(const char* identifier,void* value);

extern void *mlpackGetGaussianKernelPtr(const char* identifier);

extern void mlpacktest_go_binding();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
