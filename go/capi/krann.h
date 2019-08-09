#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetRANNModelPtr(const char* identifier,void* value);

extern void *mlpackGetRANNModelPtr(const char* identifier);

extern void mlpackkrann();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
