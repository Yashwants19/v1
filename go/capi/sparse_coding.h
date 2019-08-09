#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetSparseCodingPtr(const char* identifier,void* value);

extern void *mlpackGetSparseCodingPtr(const char* identifier);

extern void mlpacksparse_coding();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
