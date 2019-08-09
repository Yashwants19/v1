#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetKNNModelPtr(const char* identifier,void* value);

extern void *mlpackGetKNNModelPtr(const char* identifier);

extern void mlpackknn();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
