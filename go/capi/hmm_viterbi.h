#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetHMMModelPtr(const char* identifier,void* value);

extern void *mlpackGetHMMModelPtr(const char* identifier);

extern void mlpackhmm_viterbi();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
