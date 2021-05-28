#ifndef __COMMON_H
#define __COMMON_H
#include <vector>
#include "rng.h"
#include <cfloat>

/**************************
   Start of user settings
***************************/
#define SAVED_VALUES_PREC 4

#define DECAY_P_TAU 1.
#define DECAY_P_KAPPA .8
#define DECAY_Q_TAU 1.
#define DECAY_Q_KAPPA 1.

#define ITER_THRESHOLD 0.1
#define LENIENT_DELTA_THRESHOLD 5e-2
#define DELTA_THRESHOLD 1e-3
#define STRICT_DELTA_THRESHOLD 1e-4

#define SIGMOID_TABLE_SIZE 10000
#define SIGMOID_ARGBOUND 14.

#ifndef DEBUG_PRINT
    #define PRINT_INTERVAL 50
#else
    #define PRINT_INTERVAL 1
#endif

#define SAVE_INTERVAL 100

#define BED_READ_SIZE_BYTES 4096

// #define FULL_SE
// #define BLOCK_SE_APPROX
// #define SNPWISE_SE_APPROX
#define SNPWISE_SE_NEUMANN_APPROX

#define FAST_SIGMOID

// #define PRECISION_DOUBLE
#define PRECISION_FLOAT

#define NEUMANN_APPROX_NUM_ITER 6

/**************************
    End of user settings
***************************/

#define MAX_K 20

#ifdef PRECISION_DOUBLE
#define MESA_NUMERIC_TYPE double
#define ZERO DBL_MIN
#define ONE -ZERO+1
#define ENTRY_SIZE 8
#endif

#ifdef PRECISION_FLOAT
#define MESA_NUMERIC_TYPE float
#define ZERO FLT_MIN
#define ONE -ZERO+1
#define ENTRY_SIZE 4
#endif

typedef std::vector<MESA_NUMERIC_TYPE> CovRow;
typedef CovRow* CovData;
typedef uint8_t* GenoData;

class InvalidGenoFileError : std::exception {
    public: const char* what() const noexcept { return "Invalid genotype file"; }
};

enum GenoFileT {
    tsv,
    ttsv,
    bed
};


#endif