#ifndef __TSID_DEPRECATED_HH__
#define __TSID_DEPRECATED_HH__

#if defined(__GNUC__) || defined(__clang__)
#define TSID_DEPRECATED __attribute__((deprecated))
#else
#define TSID_DEPRECATED
#endif

#endif  // __TSID_DEPRECATED_HH__
