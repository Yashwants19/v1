#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetHMMModelPtr(const char* identifier, void* value);

extern void *mlpackGetHMMModelPtr(const char* identifier);

extern void mlpackHmmGenerate();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
