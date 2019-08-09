#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetRSModelPtr(const char* identifier,void* value);

extern void *mlpackGetRSModelPtr(const char* identifier);

extern void mlpackrange_search();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
