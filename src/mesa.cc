#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <iterator>
#include <algorithm>
#include <exception>
#include <iomanip>
#include <limits>
#include <cstddef>
#include <math.h>
#include "mesa.h"
#include "common.h"
#include "utils.h"
#include <omp.h>
#include <iomanip>
#include <chrono>
#include <stdexcept>

using std::sqrt;
using std::isnan;
using std::isfinite;

std::chrono::high_resolution_clock::time_point t0_tp, t1_tp, t2_tp;
std::time_t t0, t1, t2;

#ifdef DEBUG_PRINT
RNG shared_rng = RNG(1234);
#else
RNG shared_rng = RNG();
#endif

MESA_NUMERIC_TYPE *pl, *old_pl;

#ifdef FAST_SIGMOID
MESA_NUMERIC_TYPE SIGMOID_TBL[SIGMOID_TABLE_SIZE];
void init_sigmoid_tbl() {
    for (size_t i = 0; i < SIGMOID_TABLE_SIZE; ++i) {
        SIGMOID_TBL[i] = std::exp( -SIGMOID_ARGBOUND + SIGMOID_ARGBOUND*2./(SIGMOID_TABLE_SIZE - 1)*i );
        SIGMOID_TBL[i] /= (SIGMOID_TBL[i] + 1.);
    }
}

inline MESA_NUMERIC_TYPE
sigmoid(MESA_NUMERIC_TYPE arg) {
    return SIGMOID_TBL[(int)((clip(arg, (MESA_NUMERIC_TYPE)-SIGMOID_ARGBOUND, (MESA_NUMERIC_TYPE)SIGMOID_ARGBOUND)+SIGMOID_ARGBOUND)/(SIGMOID_ARGBOUND * 2)*(SIGMOID_TABLE_SIZE - 1))];
}

#else

inline MESA_NUMERIC_TYPE
sigmoid(MESA_NUMERIC_TYPE arg) {
    arg = std::exp(arg);
    return arg / (arg + 1.);
    // return std::tanh(arg/2)/2 + 1./2;
}
#endif


MESA_NUMERIC_TYPE *t_pl, *t_old_pl;

constexpr MESA_NUMERIC_TYPE prec_converter = pow(10, SAVED_VALUES_PREC);
constexpr MESA_NUMERIC_TYPE prec_inverter = 1./prec_converter;

MESA_MATRIX_TYPE* acc;
MESA_MATRIX_TYPE* R;
MESA_MATRIX_TYPE* M;
MESA_MATRIX_TYPE q_phi_grad2;
MESA_VECTOR_TYPE q_grad2;
MESA_VECTOR_TYPE buffer, result;
#if defined(FULL_SE) || defined(BLOCK_SE_APPROX)
#pragma omp threadprivate (t_pl,t_old_pl)
#endif

#ifdef SNPWISE_SE_APPROX
#pragma omp threadprivate (t_pl,t_old_pl,acc)
#endif

#ifdef SNPWISE_SE_NEUMANN_APPROX
#pragma omp threadprivate (t_pl,t_old_pl,acc,R,M)
#endif

MESA::MESA(GenoDataReader *gdr, CovDataReader *cdr, Config *config) : config{config}
{
    N = config->get_N();
    K = config->get_K();
    G = config->get_G();

    #pragma omp parallel
    {
        t_pl = new MESA_NUMERIC_TYPE[K];
        t_old_pl = new MESA_NUMERIC_TYPE[K*3];
    }
    pl = new MESA_NUMERIC_TYPE[K];
    old_pl = new MESA_NUMERIC_TYPE[K];

    if (N != gdr->shape[0])
        throw std::exception();
    if (G != gdr->shape[1])
        throw std::exception();
    if (cdr->shape[0] > 0 && cdr->shape[0] != N)
        throw std::exception();

    cov_num = cdr->shape[1];
    if (cov_num)
        std::cout << "cov_num: " << cov_num << std::endl;
    genodata = gdr->get_data();
    
    std::cout << "genodata (i=0 to 5 and l=0 to 5):\n";
    for (size_t i = 0; i < 6; ++i) {
        for (size_t l = 0; l < 6; ++l) {
            std::cout << (int)genodata[i * G + l];
            std::cout << (l < 5 ? "\t" : "\n");
        }
    }
    std::cout << std::endl;

    covdata = cdr->get_data();

    std::cout << "covariates (i=0 to 5):\n";
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < cov_num; ++j) {
            std::cout << covdata[i][j];
            std::cout << (j < cov_num - 1 ? "\t" : "\n");
        }
    }
    std::cout << std::endl;

    P_params = MESA_VECTOR_TYPE::Zero(K * G * (cov_num + 1));
    effect_se_ptr = new MESA_NUMERIC_TYPE[K * G * cov_num]();
    std::cout << "K * G: " << K * G << "\n";
    std::cout << "P_params size: " << P_params.size() << "\n";
    Q = new MESA_NUMERIC_TYPE[N * K]();
    Q_buffer = new MESA_NUMERIC_TYPE[N * K]();

    init_params();

    l_ind_vec.resize(G);
    std::iota(l_ind_vec.begin(), l_ind_vec.end(), 0);

    #ifndef DEBUG_PRINT
    std::cout << "Sorting individuals indices..." << std::endl;
    shared_rng.shuffle(l_ind_vec);
    std::cout << "Finished sorting." << std::endl;
    #endif
    
    snp_batch_size = std::min(G, (size_t)config->get_batch_size());
    not_converged_snps.resize(snp_batch_size);
    globally_not_converged_snps.resize(snp_batch_size);
    std::cout << "snp_batch_size: " << snp_batch_size << std::endl;
    
    not_converged_subjects.resize(N);

    buffer = MESA_VECTOR_TYPE::Zero(K * snp_batch_size * (cov_num + 1));
    result = MESA_VECTOR_TYPE::Zero(buffer.size());

    #ifdef FULL_SE
    acc = new MESA_MATRIX_TYPE();
    *acc = MESA_MATRIX_TYPE::Zero(G * (cov_num + 1), K * G * (cov_num + 1));
    q_grad2 = MESA_VECTOR_TYPE::Zero(N * K);
    q_phi_grad2 = MESA_MATRIX_TYPE::Zero(N * K, G * (cov_num + 1));
    #endif

    #ifdef BLOCK_SE_APPROX
    acc = new MESA_MATRIX_TYPE();
    *acc = MESA_MATRIX_TYPE::Zero(snp_batch_size * (cov_num + 1), snp_batch_size * K * (cov_num + 1));
    q_grad2 = MESA_VECTOR_TYPE::Zero(N * K);
    q_phi_grad2 = MESA_MATRIX_TYPE::Zero(N * K, snp_batch_size * (cov_num + 1));
    #endif
    
    #ifdef SNPWISE_SE_APPROX
    q_grad2 = MESA_VECTOR_TYPE::Zero(N * K);
    q_phi_grad2 = MESA_MATRIX_TYPE::Zero(N * K, snp_batch_size * (cov_num + 1));
    #pragma omp parallel
    {
        acc = new MESA_MATRIX_TYPE();
        *acc = MESA_MATRIX_TYPE::Zero(cov_num + 1, (cov_num + 1) * K);
    }
    #endif
    
    #ifdef SNPWISE_SE_NEUMANN_APPROX
    q_grad2 = MESA_VECTOR_TYPE::Zero(N * K);
    q_phi_grad2 = MESA_MATRIX_TYPE::Zero(N * K, snp_batch_size * (cov_num + 1));
    #pragma omp parallel
    {
        acc = new MESA_MATRIX_TYPE();
        *acc = MESA_MATRIX_TYPE::Zero(cov_num + 1, (cov_num + 1) * K);
        R = new MESA_MATRIX_TYPE();
        *R = MESA_MATRIX_TYPE::Zero(cov_num + 1, (cov_num + 1) * K);
        M = new MESA_MATRIX_TYPE();
        *M = MESA_MATRIX_TYPE::Zero(cov_num + 1, (cov_num + 1) * K);
    }
    #endif
    sqrtN = std::sqrt(N);
    inv_sqrtN = 1. / sqrtN;
    inv_N = 1. / N;
    sqrtG = std::sqrt(G);
    lg2G = log2(G);
}

void
MESA::init_params() {   
    size_t offset;

    for (size_t l = 0; l < G; ++l)
    {
        for (size_t k = 0; k < K; ++k)
        {
            offset = (k * G + l) * (cov_num + 1);

            // P_params[offset] = shared_rng.uniform(0., SIGMOID_ARGBOUND/2.);
            // P_params[offset] = shared_rng.uniform(-3., 3.);
            P_params[offset] = shared_rng.uniform(0., 3.);
            
            for (size_t j = 1; j <= cov_num; ++j)
            {
                P_params[offset + j] = shared_rng.uniform(-.5, .5);
            }
        }
    }
    
    MESA_NUMERIC_TYPE tmp = 1.0 / K;
    
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < K; ++k) {
            Q[i * K + k] = tmp;
        }
    }
}

MESA::~MESA()
{
    if (effect_se_ptr != nullptr)
        delete[] effect_se_ptr;
    delete[] Q;
    delete[] Q_buffer;
    delete[] pl;
    delete[] old_pl;
}

void
MESA::load_est(ResultDataReader& rdr)
{
    size_t offset;
    size_t cov_num = rdr.get_cov_num();
    const MESA_NUMERIC_TYPE* est_theta = rdr.get_data();

    if (est_theta != nullptr) {
        std::cout << "cov_num: " << cov_num << "\n";
        std::cout << "Reading estimated parameters..." << std::endl;
        
        for (size_t l = 0; l < G; ++l)
        {
            for (size_t k = 0; k < K; ++k) {
                offset = (k * G + l) * (cov_num + 1);
                P_params[offset] = est_theta[offset];

                for (size_t j = 1; j <= cov_num; ++j)
                {
                    P_params[offset + j] = est_theta[offset + j];
                }
            }
        }

        offset = K * G * (cov_num + 1);
        const MESA_NUMERIC_TYPE rounding_fix = 5 * exp10(-SAVED_VALUES_PREC - 1);
        
        for (size_t i = 0; i < N; ++i)
        {
            MESA_NUMERIC_TYPE norm_const = rounding_fix * K;
            for (size_t k = 0; k < K; ++k) {
                Q[i * K + k] = est_theta[offset + i * K + k] + rounding_fix;
                norm_const += est_theta[offset + i * K + k];
            }

            for (size_t k = 0; k < K; ++k) {
                Q[i * K + k] /= norm_const;
            }
        }

        for (size_t l = 0; l < std::min((size_t)10, G); ++l) {
            for (size_t k = 0; k < K; ++k) {
                std::cout << P_params[(k * G + l) * (cov_num + 1)] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        if (cov_num) {
            for (size_t l = 0; l < std::min((size_t)10, G); ++l) {
                for (size_t k = 0; k < K; ++k) {
                    for (size_t j = 1; j <= cov_num; ++j) {
                        std::cout << P_params[(k * G + l) * (cov_num + 1) + j] << " ";
                    }
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }

        for (size_t i = 0; i < std::min((size_t)10, N); ++i) {
            for (size_t k = 0; k < K; ++k) {
                std::cout << Q[i * K + k] << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }
}

inline MESA_NUMERIC_TYPE
fkg(uint8_t g, MESA_NUMERIC_TYPE pkl)
{
    switch (g) {
        case 0:
            return (pkl * (pkl - 2.) + 1.);
        case 1:
            return pkl * (-pkl + 1.);
        case 2:
            return pkl * pkl;
        default:
            return 0.;
    }
}


inline MESA_NUMERIC_TYPE
fkg(uint8_t g, MESA_NUMERIC_TYPE pkl, MESA_NUMERIC_TYPE qik)
{
    switch (g) {
        case 0:
            return qik * (pkl * (pkl - 2.) + 1.);
        case 1:
            return qik * pkl * (-pkl + 1.);
        case 2:
            return qik * pkl * pkl;
        default:
            return 0.;
    }
}

inline void
MESA::print_Q_for_I(size_t i)
{
    if ((i >= 0) && (i < N))
    {
        std::cout << "Q for i=" << i << ": " << std::fixed << std::setprecision(3);
        for (size_t k = 0; k < K; ++k)
        {
            std::cout << Q[i * K + k] << (k < K - 1 ? ", " : "\n");
        }
    }
}

inline void
MESA::print_P_for_G(size_t l)
{
    if ((l >= 0) && (l < G))
    {
        std::cout << "P for l=" << l << ": " << std::fixed << std::setprecision(3);
        for (size_t k = 0; k < K; ++k)
        {
            std::cout << P_params[(k * G + l) * (cov_num + 1)] << (k < K - 1 ? ", " : "\n");
        }
    }
}

MESA_NUMERIC_TYPE
MESA::obs_logl(const MESA_VECTOR_TYPE &x)
{
    MESA_NUMERIC_TYPE pkli = 0, cumsum = 0;
    size_t abs_offset = 0;


    #pragma omp parallel for private(abs_offset, pkli) reduction(+:cumsum)
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t l = 0; l < G; ++l)
        {
            uint8_t g = genodata[i * G + l];

            if (g < 9) {
                MESA_NUMERIC_TYPE t_sum = 0;
                for (size_t k = 0; k < K; ++k)
                {
                    abs_offset = (k * G + l) * (cov_num + 1);
                    pkli = x[abs_offset];

                    for (size_t j = 0; j < cov_num; ++j) {
                        pkli += x[abs_offset + j + 1] * covdata[i][j];
                    }

                    pkli = sigmoid(pkli);
                    t_sum += fkg(g, pkli, Q[i * K + k]);
                }

                cumsum += (g == 1 ? M_LN2 : 0.) + log(t_sum);
            }
        }
    }

    return cumsum;
}

MESA_NUMERIC_TYPE
MESA::obs_logl_2(const MESA_VECTOR_TYPE &x)
{
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    MESA_NUMERIC_TYPE* obs_logl_ = new MESA_NUMERIC_TYPE[num_threads]();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t l = 0; l < G; ++l)
        {
            uint8_t g = genodata[i * G + l];
            if (g < 9) {
                MESA_NUMERIC_TYPE t_sum = 0;
                for (size_t k = 0; k < K; ++k)
                {
                    size_t abs_offset = (k * G + l) * (cov_num + 1);
                    MESA_NUMERIC_TYPE pkli = x[abs_offset];

                    for (size_t j = 0; j < cov_num; ++j) {
                        pkli += x[abs_offset + j + 1] * covdata[i][j];
                    }

                    pkli = sigmoid(pkli);
                    t_sum += fkg(g, pkli, Q[i * K + k]);
                }

                // std::cout << omp_get_thread_num() << std::endl;
                obs_logl_[omp_get_thread_num()] += (g == 1 ? M_LN2 : 0.) + log(t_sum);
            }
        }
    }

    MESA_NUMERIC_TYPE result = 0;

    for (auto x = 0; x < num_threads; ++x) {
        result += obs_logl_[x];
    }

    delete[] obs_logl_;

    return result;
}

MESA_NUMERIC_TYPE
MESA::subset_obs_logl(const MESA_VECTOR_TYPE &x, const std::vector<size_t> &snp_batch)
{
    MESA_NUMERIC_TYPE pkli = 0, cumsum = 0;
    size_t rel_offset;


    #pragma omp parallel for private(rel_offset, pkli) reduction(+:cumsum)
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t rel_l = 0; rel_l < snp_batch.size(); ++rel_l)
        {
            uint8_t g = genodata[i * G + snp_batch[rel_l]];
            if (g < 9) {
                MESA_NUMERIC_TYPE t_sum = 0;
                for (size_t k = 0; k < K; ++k)
                {
                    rel_offset = (k * snp_batch.size() + rel_l) * (cov_num + 1);
                    pkli = x[rel_offset];

                    for (size_t j = 0; j < cov_num; ++j) {
                        pkli += x[rel_offset + j + 1] * covdata[i][j];
                    }

                    pkli = sigmoid(pkli);
                    t_sum += fkg(g, pkli, Q[i * K + k]);
                }

                cumsum += (g == 1 ? M_LN2 : 0.) + log(t_sum);
            }
        }
    }

    return cumsum;
}

void
MESA::pretrain() {
    std::cout << "Start pre-training parameters with R=0" << std::endl;
    t1_tp = std::chrono::high_resolution_clock::now();
    t1 = std::chrono::high_resolution_clock::to_time_t(t1_tp);
    std::cout << "Pre-training starts at: " << std::put_time(std::localtime(&t1), "%F %T") << std::endl;
    MESA_NUMERIC_TYPE delta_P_params, delta_Q;
    
    size_t it = 1;
    
    std::vector<size_t> snp_batch;
    snp_batch.reserve(snp_batch_size);
    shared_rng.sample_snps(l_ind_vec.begin(), l_ind_vec.end(), std::back_inserter(snp_batch), snp_batch_size);

    MESA_VECTOR_TYPE old_P_params = MESA_VECTOR_TYPE::Zero(K * snp_batch_size);
    MESA_VECTOR_TYPE P_params_norm_const = MESA_VECTOR_TYPE::Zero(old_P_params.size());
    MESA_VECTOR_TYPE new_P_params = MESA_VECTOR_TYPE::Zero(old_P_params.size());
    MESA_VECTOR_TYPE old_Q = MESA_VECTOR_TYPE::Zero(N * K);
    MESA_VECTOR_TYPE new_Q = MESA_VECTOR_TYPE::Zero(old_Q.size());

    for (size_t rel_l = 0; rel_l < snp_batch_size; ++rel_l)
    {
        not_converged_snps[rel_l] = true;
        for (size_t k = 0; k < K; ++k) {
            old_P_params[k * snp_batch_size + rel_l] = sigmoid(P_params[(k * G + snp_batch[rel_l]) * (cov_num + 1)]);
        }
    }

    for (size_t i = 0; i < N; ++i) {
        not_converged_subjects[i] = true;
        for (size_t k = 0; k < K; ++k) {
            old_Q[i * K + k] = Q[i * K + k];
        }
    }

    MESA_NUMERIC_TYPE lr;
    const MESA_NUMERIC_TYPE sigmoid_lb = sigmoid(-SIGMOID_ARGBOUND);
    const MESA_NUMERIC_TYPE sigmoid_ub = sigmoid(SIGMOID_ARGBOUND);
    const MESA_NUMERIC_TYPE inv_snp_batch_size = 1.0 / snp_batch_size;

    while (true) {
        #pragma omp parallel for
        for (size_t rel_l = 0; rel_l < snp_batch_size; ++rel_l) {
            if (not_converged_snps[rel_l])
            {
                bool not_converged = false;
                for (size_t i = 0; i < N; ++i) {
                    uint8_t g = genodata[i * G + snp_batch[rel_l]];
                    if (g < 9) {
                        MESA_NUMERIC_TYPE norm_const = 0;

                        for (size_t k = 0; k < K; ++k) {
                            t_old_pl[k] = fkg(g, old_P_params[k * snp_batch_size + rel_l], old_Q[i * K + k]);
                            norm_const += t_old_pl[k];
                        }
                        
                        for (size_t k = 0; k < K; ++k) {
                            MESA_NUMERIC_TYPE multiplier = t_old_pl[k] / norm_const;
                            new_P_params[k * snp_batch_size + rel_l] += multiplier * g;
                            P_params_norm_const[k * snp_batch_size + rel_l] += multiplier;
                        } 
                    } else {
                        for (size_t k = 0; k < K; ++k) {
                            t_old_pl[k*3] = (1-old_P_params[k * snp_batch_size + rel_l])*(1-old_P_params[k * snp_batch_size + rel_l])*old_Q[i * K + k];
                            t_old_pl[k*3+1] = 2.*(1-old_P_params[k * snp_batch_size + rel_l])*old_P_params[k * snp_batch_size + rel_l]*old_Q[i * K + k];
                            t_old_pl[k*3+2] = old_P_params[k * snp_batch_size + rel_l]*old_P_params[k * snp_batch_size + rel_l]*old_Q[i * K + k];
                        }

                        for (size_t k = 0; k < K; ++k) {
                            for (uint8_t g = 0; g <= 2; ++g) {
                                new_P_params[k * snp_batch_size + rel_l] += t_old_pl[k*3+g] * g;
                                P_params_norm_const[k * snp_batch_size + rel_l] += t_old_pl[k*3+g];
                            }
                        }
                    }
                }

                for (size_t k = 0; k < K; ++k) {
                    size_t rel_offset = k * snp_batch_size + rel_l;
                    new_P_params[rel_offset] /= 2 * P_params_norm_const[rel_offset];    
                    MESA_NUMERIC_TYPE diff = abs(new_P_params[rel_offset] - old_P_params[rel_offset]);
                    not_converged = not_converged || (diff >= STRICT_DELTA_THRESHOLD);
                }
                not_converged_snps[rel_l] = not_converged;
            }
        }

        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            if (not_converged_subjects[i])
            {
                bool not_converged = false;
                for (size_t rel_l = 0; rel_l < snp_batch_size; ++rel_l) {
                    uint8_t g = genodata[i * G + snp_batch[rel_l]];

                    if (g < 9) {
                        MESA_NUMERIC_TYPE norm_const = 0;

                        for (size_t k = 0; k < K; ++k) {                    
                            t_pl[k] = fkg(g, new_P_params[k * snp_batch_size + rel_l], old_Q[i * K + k]);
                            norm_const += t_pl[k];
                        }
                        
                        for (size_t k = 0; k < K; ++k) {
                            new_Q[i * K + k] += t_pl[k] / norm_const;
                        }
                    } else {                        
                        for (size_t k = 0; k < K; ++k) {
                            new_Q[i * K + k] += old_Q[i * K + k];
                        }
                    }
                }

                for(size_t k = 0; k < K; ++k) {
                    new_Q[i * K + k] *= inv_snp_batch_size;
                    size_t abs_offset = i * K + k;
                    MESA_NUMERIC_TYPE diff = abs(new_Q[abs_offset] - old_Q[abs_offset]);
                    not_converged = not_converged || (diff >= STRICT_DELTA_THRESHOLD);
                }
                not_converged_subjects[i] = not_converged;
            }
        }

        new_P_params = old_P_params - new_P_params;
        new_Q = old_Q - new_Q;
        size_t old_P_params_nonzero_count = (new_P_params.array() != 0.).count();
        size_t old_Q_nonzero_count = (new_Q.array() != 0.).count();

        if (old_P_params_nonzero_count) {
            delta_P_params = std::sqrt(new_P_params.squaredNorm() / old_P_params_nonzero_count);
        } else {
            delta_P_params = 0.;
        }        
        
        if (old_Q_nonzero_count) {
            delta_Q = std::sqrt(new_Q.squaredNorm() / old_Q_nonzero_count);
        } else {
            delta_Q = 0.;
        }
        
        new_Q = old_Q - new_Q;
        old_Q = new_Q;

        if ( delta_P_params < STRICT_DELTA_THRESHOLD && delta_Q < STRICT_DELTA_THRESHOLD ) break;

        new_P_params = old_P_params - new_P_params;
        old_P_params = new_P_params;

        for (size_t rel_l = 0; rel_l < snp_batch_size; ++rel_l) {
            if (not_converged_snps[rel_l]) {
                for (size_t k = 0; k < K; ++k) {
                    new_P_params[k * snp_batch_size + rel_l] = P_params_norm_const[k * snp_batch_size + rel_l] = 0.;
                }
            }
        }

        for (size_t i = 0; i < N; ++i) {
            if (not_converged_subjects[i]) {
                for (size_t k = 0; k < K; ++k) {
                    new_Q[i * K + k] = 0.;
                }
            }
        }
        
        if ((it++) % 50 == 0) {
            std::cout << "delta_P_params: " << std::fixed << std::setprecision(5) << delta_P_params << ", "
                      << "delta_Q: " << std::setprecision(6) << delta_Q << std::endl;
        }
        
    }

    std::cout << "delta_P_params: " << std::fixed << std::setprecision(5) << delta_P_params << ", "
              << "delta_Q: " << std::setprecision(6) << delta_Q << std::endl;
    
    for (size_t r = 0; r < N * K; ++r) {
        new_Q[r] = Q[r] = old_Q[r];
    }
        
    old_P_params = MESA_VECTOR_TYPE::Zero(K * G);
    P_params_norm_const = MESA_VECTOR_TYPE::Zero(old_P_params.size());
    new_P_params = MESA_VECTOR_TYPE::Zero(old_P_params.size());
    
    MESA_NUMERIC_TYPE rms;
    size_t rms_n;
    for (size_t x = 0; x < K * G; ++x)
    {
        // old_P_params[x] = sigmoid(P_params[x * (cov_num + 1)]);
        old_P_params[x] = sigmoid(abs( P_params[x * (cov_num + 1)]) );
    }

    std::vector<bool> not_converged_snps(G, true);
    std::cout << "Pretraining P_params..." << std::endl;

    #ifdef DEBUG_PRINT
    size_t number_of_snps_to_go_through = N + (N % snp_batch_size);
    #else
    size_t number_of_snps_to_go_through = G;
    #endif

    it = 1;
    do {
        rms = 0.;
        rms_n = 0;
        #pragma omp parallel for reduction(+:rms) reduction(+:rms_n)
        for (size_t l = 0; l < number_of_snps_to_go_through; ++l) {
            if (not_converged_snps[l]) {
                bool converged = true;

                for (size_t i = 0; i < N; ++i) {
                    uint8_t g = genodata[i * G + l];

                    if (g < 9) {
                        MESA_NUMERIC_TYPE norm_const = 0;

                        for (size_t k = 0; k < K; ++k) {
                            t_pl[k] = fkg(g, old_P_params[k * G + l], Q[i * K + k]);
                            norm_const += t_pl[k];
                        }
                        
                        for (size_t k = 0; k < K; ++k) {
                            MESA_NUMERIC_TYPE multiplier = t_pl[k] / norm_const;
                            new_P_params[k * G + l] += multiplier * g;
                            P_params_norm_const[k * G + l] += multiplier;
                        } 
                    } else {
                        for (size_t k = 0; k < K; ++k) {
                            t_old_pl[k*3] = (1-old_P_params[k * G + l])*(1-old_P_params[k * G + l])*old_Q[i * K + k];
                            t_old_pl[k*3+1] = 2.*(1-old_P_params[k * G + l])*old_P_params[k * G + l]*old_Q[i * K + k];
                            t_old_pl[k*3+2] = old_P_params[k * G + l]*old_P_params[k * G + l]*old_Q[i * K + k];
                        }
                        
                        for (size_t k = 0; k < K; ++k) {
                            for (uint8_t g = 0; g <= 2; ++g) {
                                new_P_params[k * G + l] += t_old_pl[k*3+g] * g;
                                P_params_norm_const[k * G + l] += t_old_pl[k*3+g];
                            }
                        } 
                    }
                }

                for (size_t k = 0; k < K; ++k) {
                    new_P_params[k * G + l] /= P_params_norm_const[k * G + l] * 2;
                    MESA_NUMERIC_TYPE diff = new_P_params[k * G + l] - old_P_params[k * G + l];
                    converged = converged && (diff < DELTA_THRESHOLD);
                    rms += diff * diff;
                    rms_n += 1;
                    old_P_params[k * G + l] = new_P_params[k * G + l];
                    new_P_params[k * G + l] = P_params_norm_const[k * G + l] = 0.;
                }

                not_converged_snps[l] = !converged;
            }
        }

        snp_batch.clear();
        shared_rng.sample_snps(l_ind_vec.begin(), l_ind_vec.end(), std::back_inserter(snp_batch), snp_batch_size);
        lr = std::pow(DECAY_Q_TAU + it, -DECAY_Q_KAPPA);

        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            for (size_t k = 0; k < K; ++k) {
                old_Q[i * K + k] = new_Q[i * K + k];
                new_Q[i * K + k] *= 1 - lr;
            }
            for (size_t rel_l = 0; rel_l < snp_batch_size; ++rel_l) {
                uint8_t g = genodata[i * G + snp_batch[rel_l]];

                if (g < 9) {
                    MESA_NUMERIC_TYPE norm_const = 0;

                    for (size_t k = 0; k < K; ++k) {
                        t_pl[k] = fkg(g, old_P_params[k * G + snp_batch[rel_l]], old_Q[i * K + k]);
                        norm_const += t_pl[k];
                    }
                    
                    for (size_t k = 0; k < K; ++k) {
                        new_Q[i * K + k] += lr * t_pl[k] / norm_const;
                    }
                } else {                        
                    for (size_t k = 0; k < K; ++k) {
                        new_Q[i * K + k] += lr * old_Q[i * K + k];
                    }
                }
            }
            for (size_t k = 0; k < K; ++k) {
                new_Q[i * K + k] *= inv_snp_batch_size * G / (1 - lr + lr * G);
            }
        }

        // for (size_t k = 0; k < K; ++k) {
        //     std::cout << new_Q[k] << " ";
        // }
        // std::cout << std::endl;

        if (rms_n == 0) break;
        rms = std::sqrt(rms / rms_n);
        
        if ((it++) % 50 == 0)
            std::cout << "rms (P_params): " << std::fixed << std::setprecision(6) << rms << std::endl;
    } while (rms >= DELTA_THRESHOLD);

    for (size_t r = 0; r < N * K; ++r) {
        Q[r] = new_Q[r];
    }
    
    for (size_t l = 0; l < number_of_snps_to_go_through; ++l) {
        for (size_t k = 0; k < K; ++k)
        {
            size_t ofs1 = k * G + l;
            size_t ofs2 = ofs1 * (cov_num + 1);
            old_P_params[ofs1] = clip(old_P_params[ofs1], sigmoid_lb, sigmoid_ub);
            // P_params[ofs2] = log(old_P_params[ofs1] / ( 1.0 - old_P_params[ofs1] ));
            P_params[ofs2] = abs( log(old_P_params[ofs1] / ( 1.0 - old_P_params[ofs1] )) );
        }
    }

    save_params(config->get_output_path());
    std::cout << "Finished pretraining and saved pretrained parameters." << std::endl;
}

void MESA::train()
{
    #ifndef DEBUG_PRINT
    shared_rng.shuffle(l_ind_vec);
    #endif
    
    MESA_NUMERIC_TYPE curr_obs_logl = 0.0;
    MESA_NUMERIC_TYPE delta_Q;
    
    size_t it = 1, idx;

    std::vector<size_t> snp_batch(snp_batch_size);

    MESA_VECTOR_TYPE old_P_params = MESA_VECTOR_TYPE::Zero(K * snp_batch_size * (cov_num + 1));
    MESA_VECTOR_TYPE new_P_params = MESA_VECTOR_TYPE::Zero(old_P_params.size());
    MESA_VECTOR_TYPE old_Q = MESA_VECTOR_TYPE::Zero(N * K);
    MESA_VECTOR_TYPE new_Q = MESA_VECTOR_TYPE::Zero(old_Q.size());

    for (idx = 0; idx < snp_batch_size; ++idx)
        snp_batch[idx] = l_ind_vec[((it - 1) * snp_batch_size + idx) % G];

    for (size_t i = 0; i < N; ++i) {
        not_converged_subjects[i] = true;
        for (size_t k = 0; k < K; ++k) {
            old_Q[i * K + k] = Q[i * K + k];
        }
    }

    MESA_NUMERIC_TYPE **hkgil_by_il = new MESA_NUMERIC_TYPE *[N * snp_batch_size]();
    for (size_t r = 0; r < N * snp_batch_size; ++r)
        hkgil_by_il[r] = new MESA_NUMERIC_TYPE[K]();

    MESA_VECTOR_TYPE grad1 = MESA_VECTOR_TYPE::Zero(K * snp_batch_size * (cov_num + 1));
    MESA_MATRIX_TYPE grad2 = MESA_MATRIX_TYPE::Zero(cov_num + 1, K * snp_batch_size * (cov_num + 1));
    
    t1_tp = std::chrono::high_resolution_clock::now();
    t1 = std::chrono::high_resolution_clock::to_time_t(t1_tp);
    std::cout << "Training starts at: " << std::put_time(std::localtime(&t1), "%F %T") << std::endl;

    // size_t metricsk = K * G * (cov_num + 1) + N * K, metrics_n = N;
    // size_t repeats = G / snp_batch_size + (G % snp_batch_size > 0);
    size_t snp_batch_first = idx % G;

    curr_obs_logl = 0.;

    while (true)
    {
        update(snp_batch, it, grad1, grad2, hkgil_by_il);

        for (size_t r = 0; r < N * K; ++r)
            new_Q[r] = Q[r];

        delta_Q = std::sqrt((new_Q - old_Q).squaredNorm() / new_Q.size());

        // iter_mod_save = (it - 1) % SAVE_INTERVAL;

        // metrics[iter_mod_save * 5] = curr_obs_logl;
        // metrics[iter_mod_save * 5 + 1] = metricsk * log(metrics_n) - 2 * curr_obs_logl;
        // metrics[iter_mod_save * 5 + 2] = metricsk * log(metrics_n * G) - 2 * curr_obs_logl;
        // metrics[iter_mod_save * 5 + 3] = delta_P_params;
        // metrics[iter_mod_save * 5 + 4] = delta_Q;

        for (auto r = 0; r < new_Q.size(); ++r)
            old_Q[r] = new_Q[r];

        if (it % PRINT_INTERVAL == 0)
        {
            std::cout << " >>>>  Iter " << it << "  <<<< \n";
            for (size_t i = 0; i < 3; ++i)
            {
                print_P_for_G(i);
                print_Q_for_I(i);
                std::cout << "\n";
            }
            std::cout << "Obs. logl: " << std::setprecision(5) << curr_obs_logl << ", "
                    //   << "Delta P_params (rms): " << std::setprecision(5) << delta_P_params << ", "
                      << "Delta Q (rms): " << std::setprecision(5) << delta_Q << "\n";
            
            t2_tp = std::chrono::high_resolution_clock::now();
            t2 = std::chrono::high_resolution_clock::to_time_t(t2_tp);
            std::cout << "Time now: " << std::put_time(std::localtime(&t2), "%F %T") << ", "
                      << "Time elapsed: " << std::chrono::duration_cast<std::chrono::hours>(t2_tp - t1_tp).count() << " h "
                      << std::chrono::duration_cast<std::chrono::minutes>(t2_tp - t1_tp).count() % 60 << " mins\n" << std::endl;

        }

        #ifdef DEBUG_PRINT
        // if (it++ * snp_batch_size >= N) break;
        break;
        #endif

        if (it++ * snp_batch_size >= G * config->get_epochs()) break;
        
        for (idx = 0; idx < snp_batch_size; ++idx) {
            snp_batch[idx] = l_ind_vec[(snp_batch_first + idx) % G];
        }
    
        snp_batch_first = (snp_batch_first + idx) % G;
    }
    
    save_params(config->get_output_path());

    for (size_t r = 0; r < N * snp_batch_size; ++r)
        delete[] hkgil_by_il[r];
    delete[] hkgil_by_il;
    hkgil_by_il = nullptr;

    std::cout << "Last iteration = " << it - 1 << "\n";
    for (size_t i = 0; i < 3; ++i)
    {
        print_P_for_G(i);
        print_Q_for_I(i);
        std::cout << "\n";
    }
    #ifdef DEBUG_PRINT
    curr_obs_logl = 0;
    #else
    curr_obs_logl = obs_logl_2(P_params);
    #endif
    // curr_obs_logl = 0;
    std::cout << "Obs. logl: " << std::setprecision(5) << curr_obs_logl << "\n";
}


void MESA::update(const std::vector<size_t>& snp_batch, size_t global_iter, MESA_VECTOR_TYPE &grad1, MESA_MATRIX_TYPE &grad2, MESA_NUMERIC_TYPE **hkgil_by_il)
{
    #ifdef DEBUG_PRINT
    auto t2a_tp = std::chrono::high_resolution_clock::now();
    #endif

    MESA_NUMERIC_TYPE curr_rms = 1.;
    size_t curr_rms_n = snp_batch_size * K * (cov_num + 1);

    size_t rel_offset, abs_offset;
    
    int nan_loci_count = 0;
    bool problematic_locus = false;

    for (size_t rel_l = 0; rel_l < snp_batch_size; ++rel_l)
    {
        globally_not_converged_snps[rel_l] = true;
    }
    
    #ifdef DEBUG_PRINT
    size_t local_min_not_converged = snp_batch_size;
    #endif

    for (size_t local_it = 0; local_it < config->get_em_iter(); ++local_it)
    // for (size_t local_it = 0;; ++local_it)
    {
        #ifdef DEBUG_PRINT
        auto t3_tp = std::chrono::high_resolution_clock::now();
        #endif
        // std::cout << "it: " << (it++) << "\n";
        for (size_t rel_l = 0; rel_l < snp_batch_size; ++rel_l)
        {
            not_converged_snps[rel_l] = globally_not_converged_snps[rel_l];
            for (size_t k = 0; k < K; ++k)
            {
                rel_offset = (k * snp_batch_size + rel_l) * (cov_num + 1);
                abs_offset = (k * G + snp_batch[rel_l]) * (cov_num + 1);

                for (size_t r = 0; r <= cov_num; ++r)
                {
                    buffer[rel_offset + r] = result[rel_offset + r] = P_params[abs_offset + r];
                }
            }
        }

        for (size_t newton_it = 0; newton_it < config->get_newton_iter(); ++newton_it)
        // for (size_t newton_it = 0;; ++newton_it)
        {
            // std::cout << "local_it: " << (local_it++) << "\n";
            nan_loci_count = 0;
            snp_wise_newton_update(buffer, snp_batch, grad1, grad2);


            #pragma omp parallel for private(rel_offset, problematic_locus) reduction(+:nan_loci_count)
            for (size_t rel_l = 0; rel_l < snp_batch_size; ++rel_l)
            {
                if ( not_converged_snps[rel_l] ) {
                    problematic_locus = false;
                    for (size_t k = 0; k < K; ++k)
                    {
                        bool is_constant_snp = false;
                        for (size_t j = 0; j <= cov_num; ++j)
                        {
                            rel_offset = (k * snp_batch_size + rel_l) * (cov_num + 1) + j;

                            if (j == 0)
                                buffer[rel_offset] = abs(buffer[rel_offset]);
                            
                            if ( !std::isfinite(buffer[rel_offset]) ) {
                                problematic_locus = true;
                                // buffer[rel_offset] = j ? shared_rng.uniform(-SIGMOID_ARGBOUND/2, SIGMOID_ARGBOUND/2) : 0.;
                                buffer[rel_offset] = 0.;
                            } else if (buffer[rel_offset] < -SIGMOID_ARGBOUND) {
                                if (j == 0) {
                                    buffer[rel_offset] = -SIGMOID_ARGBOUND;
                                    is_constant_snp = true;
                                } else if (is_constant_snp) {
                                    // buffer[rel_offset] = shared_rng.uniform(-SIGMOID_ARGBOUND/2, SIGMOID_ARGBOUND/2);
                                    buffer[rel_offset] = 0.;
                                } else {
                                    buffer[rel_offset] = -SIGMOID_ARGBOUND;
                                }                                
                            } else if (buffer[rel_offset] > SIGMOID_ARGBOUND) {
                                if (j == 0) {
                                    buffer[rel_offset] = SIGMOID_ARGBOUND;
                                    is_constant_snp = true;
                                } else if (is_constant_snp) {
                                    // buffer[rel_offset] = shared_rng.uniform(-SIGMOID_ARGBOUND/2, SIGMOID_ARGBOUND/2);
                                    buffer[rel_offset] = 0.;
                                } else {
                                    buffer[rel_offset] = SIGMOID_ARGBOUND;
                                }                               
                            }
                        }
                    }

                    if (problematic_locus) 
                        ++nan_loci_count;
                }
            }
            if (nan_loci_count == 0) {
                curr_rms = 0.;
                curr_rms_n = K * snp_batch_size * (cov_num + 1);

                #pragma omp parallel for reduction(+:curr_rms)
                for (size_t rel_l = 0; rel_l < snp_batch_size; ++rel_l) {
                    if (not_converged_snps[rel_l]) {
                        bool not_converged = false;
                        // MESA_NUMERIC_TYPE rotation_difference = 0;
                        for (size_t j = 0; j <= cov_num; ++j) {
                            
                            for (size_t k = 0; k < K; ++k) {
                                size_t rel_offset = (k * snp_batch_size + rel_l) * (cov_num + 1) + j;
                                MESA_NUMERIC_TYPE diff = abs(result[rel_offset] - buffer[rel_offset]);
                                // if (abs(result[rel_offset] - buffer[rel_offset]) > 6) {
                                //     std::cout << k << ", " << snp_batch[rel_l] <<  ", " << j << ": " << result[rel_offset] << ", " << buffer[rel_offset] << std::endl;
                                // }
                                // #ifdef DEBUG_PRINT
                                // if (snp_batch[rel_l] == 940) {
                                //     std::cout << k << ", " << snp_batch[rel_l] <<  ", " << j << ": " << result[rel_offset] << ", " << buffer[rel_offset] << std::endl;
                                // }
                                // #endif
                                result[rel_offset] = buffer[rel_offset];
                                not_converged = not_converged || ( diff >= DELTA_THRESHOLD );
                                curr_rms += std::pow(diff, 2.);
                            }
                        }
                        // not_converged_snps[rel_l] = not_converged && (rotation_difference != 0.);
                        not_converged_snps[rel_l] = not_converged;
                    }
                }
                
                if (curr_rms_n) curr_rms = std::sqrt(curr_rms / curr_rms_n);
                #ifdef DEBUG_PRINT
                std::cout << "local_curr_rms: " << curr_rms << std::endl;
                if (curr_rms < 0) std::cout << "local_curr_rms is negative.\n";
                if (std::isnan(curr_rms)) std::cout << "local_curr_rms is nan.\n";
                if (!std::isfinite(curr_rms)) std::cout << "local_curr_rms is not finite.\n";
                #endif

                if ( curr_rms < DELTA_THRESHOLD) break;
                // if ( curr_rms < LENIENT_DELTA_THRESHOLD ) break;
                // if ( curr_rms < DELTA_THRESHOLD ) break;                
            }
            
            // #ifdef DEBUG_PRINT
            // auto t6_tp = std::chrono::high_resolution_clock::now();
            // std::cout 
            //     << "Parameters fixing\nTime elapsed: " 
            //     << std::chrono::duration_cast<std::chrono::minutes>(t6_tp - t5_tp).count() % 60 << " m "
            //     << std::chrono::duration_cast<std::chrono::seconds>(t6_tp - t5_tp).count() % 60 << " s "
            //     << std::chrono::duration_cast<std::chrono::milliseconds>(t6_tp - t5_tp).count() % 1000 << " ms " << std::endl;
            // #endif
        }
        #ifdef DEBUG_PRINT
        std::cout << "Finished local Newton's update" << std::endl;
        auto t5_tp = std::chrono::high_resolution_clock::now();
        std::cout 
            << "Total time elapsed: " 
            << std::chrono::duration_cast<std::chrono::minutes>(t5_tp - t3_tp).count() % 60 << " m "
            << std::chrono::duration_cast<std::chrono::seconds>(t5_tp - t3_tp).count() % 60 << " s "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t5_tp - t3_tp).count() % 1000 << " ms \n" << std::endl;
        #endif

        curr_rms = 0.;
        curr_rms_n = K * snp_batch_size * (cov_num + 1);
        
        #pragma omp parallel for reduction(+:curr_rms)
        for (size_t rel_l = 0; rel_l < snp_batch_size; ++rel_l)
        {
            if (globally_not_converged_snps[rel_l]) {
                #ifdef DEBUG_PRINT
                if (local_it == config->get_em_iter() - 1) {
                    std::cout << snp_batch[rel_l] << ": ";
                }
                #endif
                bool not_converged = false;
                for (size_t r = 0; r <= cov_num; ++r)
                {
                    for (size_t k = 0; k < K; ++k)
                    {
                        size_t rel_offset = (k * snp_batch_size + rel_l) * (cov_num + 1) + r;
                        size_t abs_offset = (k * G + snp_batch[rel_l]) * (cov_num + 1) + r;
                        MESA_NUMERIC_TYPE diff = abs(result[rel_offset] - P_params[abs_offset]);
                        #ifdef DEBUG_PRINT
                        if (local_it == config->get_em_iter() - 1) {
                            std::cout << result[rel_offset] << "," << P_params[abs_offset] << " ";
                        }
                        #endif
                        // curr_rms_n += 1;
                        P_params[abs_offset] = result[rel_offset];
                        not_converged = not_converged || ( diff >= DELTA_THRESHOLD );
                        curr_rms += std::pow(diff, 2.);
                    }
                }
                #ifdef DEBUG_PRINT
                if (local_it == config->get_em_iter() - 1) {
                    std::cout << "\n";
                }
                #endif
                globally_not_converged_snps[rel_l] = not_converged;
            }
        }
        #ifdef DEBUG_PRINT
        if (local_it == config->get_em_iter() - 1 && local_min_not_converged < snp_batch_size) {
            std::cout << std::endl;
        }
        #endif

        if (curr_rms_n) curr_rms = std::sqrt(curr_rms / curr_rms_n);

        #ifdef DEBUG_PRINT
        std::cout << "global_curr_rms: " << curr_rms << std::endl;
        t5_tp = std::chrono::high_resolution_clock::now();
        std::cout 
            << "Total time elapsed: " 
            << std::chrono::duration_cast<std::chrono::minutes>(t5_tp - t3_tp).count() % 60 << " m "
            << std::chrono::duration_cast<std::chrono::seconds>(t5_tp - t3_tp).count() % 60 << " s "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t5_tp - t3_tp).count() % 1000 << " ms \n" << std::endl;
        
        std::cout << local_it << "," << local_min_not_converged << std::endl;
        #endif
        
        if (curr_rms < DELTA_THRESHOLD) break;
        // if (curr_rms < DELTA_THRESHOLD) break;
    }
    #ifdef DEBUG_PRINT
    std::cout << "global_curr_rms: " << curr_rms << std::endl;
    if (curr_rms < 0) std::cout << "global_curr_rms is negative.\n";
    if (std::isnan(curr_rms)) std::cout << "global_curr_rms is nan.\n";
    if (!std::isfinite(curr_rms)) std::cout << "global_curr_rms is not finite.\n";
    auto t2b_tp = std::chrono::high_resolution_clock::now();
    std::cout
            << "M-step time elapsed: " << std::chrono::duration_cast<std::chrono::hours>(t2b_tp - t2a_tp).count() << " h "
            << std::chrono::duration_cast<std::chrono::minutes>(t2b_tp - t2a_tp).count() % 60 << " m "
            << std::chrono::duration_cast<std::chrono::seconds>(t2b_tp - t2a_tp).count() % 60 << " s\n\n" << std::endl;
    #endif
    
    MESA_NUMERIC_TYPE lr = std::pow(DECAY_Q_TAU + global_iter, -DECAY_Q_KAPPA);        
    MESA_NUMERIC_TYPE inv_Q_norm_const = 1.0 / (1.0 - lr + lr * G);
    const MESA_NUMERIC_TYPE inv_snp_batch_size = 1.0 / snp_batch_size;
    
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        if (not_converged_subjects[i])
        {
            bool not_converged = false;
            MESA_NUMERIC_TYPE g, hkgil_norm_const, pkli;
            size_t abs_offset;

            for (size_t rel_l = 0; rel_l < snp_batch_size; ++rel_l)
            {
                if (genodata[i * G + snp_batch[rel_l]] < 9) {
                    g = genodata[i * G + snp_batch[rel_l]];
                    hkgil_norm_const = 0;

                    for (size_t k = 0; k < K; ++k)
                    {
                        abs_offset = (k * G + snp_batch[rel_l]) * (cov_num + 1);
                        pkli = P_params[abs_offset];
                        #ifdef DEBUG_PRINT
                        if ( !std::isfinite(pkli) ) std::cout << "(k, l, r): " << "(" << k << ", " << snp_batch[rel_l] << ", " << 0 << ") is nan\n";
                        #endif

                        for (size_t j = 0; j < cov_num; ++j) {
                            pkli += P_params[abs_offset + j + 1] * covdata[i][j];
                            #ifdef DEBUG_PRINT
                            if ( !std::isfinite(P_params[abs_offset + j + 1]) ) std::cout << "(k, l, r): " << "(" << k << ", " << snp_batch[rel_l] << ", " << j+1 << ") is nan\n";
                            #endif
                        }

                        // if (pkli > SIGMOID_ARGBOUND) {
                        //     pkli = sigmoid(SIGMOID_ARGBOUND);
                        // } else if (pkli < -SIGMOID_ARGBOUND) {
                        //     pkli = sigmoid(-SIGMOID_ARGBOUND);
                        // } else {
                        //     pkli = sigmoid(pkli);
                        // }
                        pkli = sigmoid(pkli);


                        hkgil_by_il[i * snp_batch_size + rel_l][k] = fkg(g, pkli, Q[i * K + k]);
                        hkgil_norm_const += hkgil_by_il[i * snp_batch_size + rel_l][k];
                    }

                    if ( std::isnormal(hkgil_norm_const) ) {
                        for (size_t k = 0; k < K; ++k)
                            Q_buffer[i * K + k] += hkgil_by_il[i * snp_batch_size + rel_l][k] / hkgil_norm_const * inv_snp_batch_size;
                    } else {
                        for (size_t k = 0; k < K; ++k)
                            Q_buffer[i * K + k] += Q[i * K + k] * inv_snp_batch_size;
                    }
                } else {
                    for (size_t k = 0; k < K; ++k)
                        Q_buffer[i * K + k] += Q[i * K + k] * inv_snp_batch_size;
                }
            }
            
            MESA_NUMERIC_TYPE tmp_sum = 0.;
            MESA_NUMERIC_TYPE old_Q, diff;
            for (size_t k = 0; k < K - 1; ++k)
            {
                if (std::isfinite(Q_buffer[i * K + k])) {
                    old_Q = Q[i * K + k];
                    Q_buffer[i * K + k] = clip(Q_buffer[i * K + k], ZERO, ONE);
                    Q[i * K + k] = ( Q[i * K + k] * (1.0 - lr) + Q_buffer[i * K + k] * G * lr ) * inv_Q_norm_const;
                    diff = abs(Q[i * K + k] - old_Q);
                    not_converged = not_converged || (diff >= DELTA_THRESHOLD);
                } else {
                    not_converged = true;
                }
                tmp_sum += Q[i * K + k];
                Q_buffer[i * K + k] = 0.;
            }
            Q[i * K + K - 1] = clip(-tmp_sum+1, ZERO, ONE);
            Q_buffer[i * K + K - 1] = 0.;
            not_converged_subjects[i] = not_converged;
        }
    }
}

void 
MESA::snp_wise_newton_update(MESA_VECTOR_TYPE &x, const std::vector<size_t> &snp_batch, MESA_VECTOR_TYPE &grad1, MESA_MATRIX_TYPE &grad2)
{
    grad1.setZero();
    grad2.setZero();
    MESA_MATRIX_TYPE I = MESA_MATRIX_TYPE::Identity(cov_num + 1, cov_num + 1);

    #pragma omp parallel for
    for (size_t rel_l = 0; rel_l < snp_batch_size; ++rel_l)
    {
        if (not_converged_snps[rel_l])
        {
            uint8_t g;
            MESA_NUMERIC_TYPE hkgil_norm_const, part1, part2, multiplier, c1, c2, t_old_pl_tmp;
            size_t rel_offset, abs_offset;
            
            for (size_t i = 0; i < N; ++i)
            {
                if (genodata[i * G + snp_batch[rel_l]] < 9) {
                    g = genodata[i * G + snp_batch[rel_l]];
                    hkgil_norm_const = 0.;
                    
                    for (size_t k = 0; k < K; ++k)
                    {                
                        rel_offset = (k * snp_batch.size() + rel_l) * (cov_num + 1);
                        abs_offset = (k * G + snp_batch[rel_l]) * (cov_num + 1);
                        
                        t_pl[k] = x[rel_offset];
                        t_old_pl[k] = P_params[abs_offset];

                        for (size_t j = 0; j < cov_num; ++j) {
                            t_pl[k] += x[rel_offset + j + 1] * covdata[i][j];
                            t_old_pl[k] += P_params[abs_offset + j + 1] * covdata[i][j];
                        }

                        // #ifdef DEBUG_PRINT
                        // if (snp_batch[rel_l] == 940) {
                        //     std::cout << "t_pl for k = " << k << ": " << t_pl[k] << ", " << t_old_pl[k] << "\n";
                        // }
                        // #endif

                        t_pl[k] = sigmoid(t_pl[k]);
                        t_old_pl[k] = sigmoid(t_old_pl[k]);
                        
                        // #ifdef DEBUG_PRINT
                        // if (snp_batch[rel_l] == 940) {
                        //     std::cout << "t_pl(sigmoid) for k = " << k << ": " << t_pl[k] << ", " << t_old_pl[k] << "\n";
                        // }
                        // #endif

                        t_old_pl[k] = fkg(g, t_old_pl[k], Q[i * K + k]);
                        hkgil_norm_const += t_old_pl[k];
                    }

                    for (size_t k = 0; k < K; ++k) {
                        rel_offset = (k * snp_batch.size() + rel_l) * (cov_num + 1);
                        abs_offset = (k * G + snp_batch[rel_l]) * (cov_num + 1);

                        multiplier = t_old_pl[k] / hkgil_norm_const;
                        part1 = (t_pl[k]*2. - g) * multiplier;
                        part2 = t_pl[k]*2. * (-t_pl[k] + 1.) * multiplier;
                        
                        grad1[rel_offset] += part1;
                        grad2(0, rel_offset) += part2;

                        for (size_t j1 = 1; j1 <= cov_num; ++j1)
                        {
                            c1 = covdata[i][j1 - 1];
                            grad1[rel_offset + j1] += part1 * c1;
                            grad2(j1, rel_offset) += part2 * c1;
                            grad2(0, rel_offset + j1) += part2 * c1;
                            // grad2(0, rel_offset + j1) = grad2(j1, rel_offset);
                            grad2(j1, rel_offset + j1) += part2 * c1 * c1;

                            for (size_t j2 = j1 + 1; j2 <= cov_num; ++j2)
                            {
                                c2 = covdata[i][j2 - 1];
                                grad2(j1, rel_offset + j2) += part2 * c1 * c2;
                                grad2(j2, rel_offset + j1) += part2 * c1 * c2;
                                // grad2(j2, rel_offset + j1) = grad2(j1, rel_offset + j2);
                            }
                        }
                    }
                } else {
                    for (size_t k = 0; k < K; ++k)
                    {
                        rel_offset = (k * snp_batch.size() + rel_l) * (cov_num + 1);
                        abs_offset = (k * G + snp_batch[rel_l]) * (cov_num + 1);
                        
                        t_pl[k] = x[rel_offset];
                        t_old_pl[k] = P_params[abs_offset];

                        for (size_t j = 0; j < cov_num; ++j) {
                            t_pl[k] += x[rel_offset + j + 1] * covdata[i][j];
                            t_old_pl[k] += P_params[abs_offset + j + 1] * covdata[i][j];
                        }

                        t_pl[k] = sigmoid(t_pl[k]);
                        t_old_pl[k] = sigmoid(t_old_pl[k]);
                    }

                    for (size_t k = 0; k < K; ++k) {
                        rel_offset = (k * snp_batch.size() + rel_l) * (cov_num + 1);
                        abs_offset = (k * G + snp_batch[rel_l]) * (cov_num + 1);

                        part1 = 0.;
                        part2 = 0.;

                        for (uint8_t g_ = 0; g_ <= 2; ++g_) { 
                            t_old_pl_tmp = fkg(g_, t_old_pl[k], Q[i*K+k]);
                            part1 += (t_pl[k]*2. - g_) * t_old_pl_tmp;
                            part2 += t_pl[k]*2. * (-t_pl[k] + 1.) * t_old_pl_tmp;
                        }
                        
                        grad1[rel_offset] += part1;
                        grad2(0, rel_offset) += part2;

                        for (size_t j1 = 1; j1 <= cov_num; ++j1)
                        {
                            c1 = covdata[i][j1 - 1];
                            grad1[rel_offset + j1] += part1 * c1;
                            grad2(j1, rel_offset) += part2 * c1;
                            grad2(0, rel_offset + j1) += part2 * c1;
                            // grad2(0, rel_offset + j1) = grad2(j1, rel_offset);
                            grad2(j1, rel_offset + j1) += part2 * c1 * c1;

                            for (size_t j2 = j1 + 1; j2 <= cov_num; ++j2)
                            {
                                c2 = covdata[i][j2 - 1];
                                grad2(j1, rel_offset + j2) += part2 * c1 * c2;
                                grad2(j2, rel_offset + j1) += part2 * c1 * c2;
                                // grad2(j2, rel_offset + j1) = grad2(j1, rel_offset + j2);
                            }
                        }
                    }
                }
            }
            
            for (size_t k = 0; k < K; ++k) {            
                rel_offset = (k * snp_batch_size + rel_l) * (cov_num + 1);
                if (cov_num > 0)
                {
                    // #ifdef DEBUG_PRINT
                    // if (snp_batch[rel_l] == 940) {
                    //     std::cout << "newton delta for k = " << k << ": " << grad2.block(0, rel_offset, cov_num + 1, cov_num + 1).inverse() * grad1.segment(rel_offset, cov_num + 1) << "\n";
                    // }
                    // #endif
                    // x.segment(rel_offset, cov_num + 1) -= grad2.block(0, rel_offset, cov_num + 1, cov_num + 1).inverse() * grad1.segment(rel_offset, cov_num + 1);
                    x.segment(rel_offset, cov_num + 1) -= grad2.block(0, rel_offset, cov_num + 1, cov_num + 1).llt().solve(I) * grad1.segment(rel_offset, cov_num + 1);
                }
                else
                {
                    x[rel_offset] -= grad1[rel_offset] / grad2(0, rel_offset);
                }
            }
        }
    }
}

// void
// MESA::init_metrics_file(const std::string output_path)
// {
//     std::string metrics_path = output_path + "_metrics.tsv";
//     std::ofstream ofile(metrics_path, std::ios::out | std::ios::trunc);
//     ofile.close();
// }

// void
// MESA::save_metrics(const std::string output_path, size_t save_rows) {
//     std::string metrics_path = output_path + "_metrics.tsv";
//     std::ofstream ofile(metrics_path, std::ios::out | std::ios::app);

//     for (size_t r = 0; r < save_rows; ++r)
//     {   
//         for (size_t m = 0; m < 5; ++m)
//         {
//             if (m < 4)
//                 ofile << std::fixed << std::setprecision(SAVED_VALUES_PREC);
//             else
//                 ofile << std::fixed << std::setprecision(6);
//             ofile << metrics[r * 5 + m] << (m < 4 ? "\t" : "\n");
//         } 
//     }

//     ofile.close();
// }

void
MESA::save_params(const std::string output_path)
{
    std::string p_path = output_path + "_p.tsv";
    std::string q_path = output_path + "_q.tsv";

    std::ofstream ofile(p_path, std::ios::out);

    for (size_t l = 0; l < G; ++l)
    {
        for (size_t k = 0; k < K; ++k)
        {
            ofile << std::fixed << std::setprecision(SAVED_VALUES_PREC) << P_params[(k * G + l) * (cov_num + 1)];
            ofile << (k < K - 1 ? "\t" : (l < G - 1 ? "\n" : ""));
        }
    }
    ofile.close();

    ofile.open(q_path, std::ios::out);

    if (K == 1) {
        for (size_t i = 0; i < N; ++i)
        {
            ofile << std::fixed << std::setprecision(SAVED_VALUES_PREC) << 1. << (i < N - 1 ? "\n" : "");
        }
    } else if (K > 1) {
        for (size_t i = 0; i < N; ++i) {
            MESA_NUMERIC_TYPE tmp_unity = 1.;
            for (size_t k = 0; k < K - 1; ++k)
            {
                MESA_NUMERIC_TYPE tmp = round(Q[i * K + k] * prec_converter) * prec_inverter;
                tmp_unity -= tmp;
                ofile << std::fixed << std::setprecision(SAVED_VALUES_PREC) << tmp << "\t";
            }
            tmp_unity = std::max(tmp_unity, ZERO);
            ofile << std::fixed << std::setprecision(SAVED_VALUES_PREC) << tmp_unity << (i < N - 1 ? "\n" : "");
        }
    }
    ofile.close();

    size_t offset = 0;

    if (cov_num > 0)
    {
        std::string effect_path = output_path + "_effect.tsv";
        ofile.open(effect_path, std::ios::out);

        for (size_t l = 0; l < G; ++l)
        {
            for (size_t k = 0; k < K; ++k)
            {
                offset = (k * G + l) * (cov_num + 1);
                for (size_t j = 1; j <= cov_num; ++j)
                {
                    ofile << std::fixed << std::setprecision(SAVED_VALUES_PREC) << P_params[offset + j];
                    ofile << ((k < K - 1 || j < cov_num) ? "\t" : (l < G - 1 ? "\n" : ""));
                }
            }
        }
        ofile.close();
    }
}

void
MESA::effect_se_full_approx() {
    if (G > snp_batch_size) {
        std::vector<size_t> snp_batch(snp_batch_size);
        size_t num_batches = G / snp_batch_size;
        size_t last_batch_size = G % snp_batch_size;

        for (size_t b = 0; b < num_batches; ++b) {
            std::iota(snp_batch.begin(), snp_batch.end(), b * snp_batch_size);
            // for (size_t rel_l = 0; rel_l < snp_batch_size; ++rel_l) {
            //     snp_batch[rel_l] = l_ind_vec[b * snp_batch_size + rel_l];
            // }
            effect_se_approx(snp_batch);
        }

        if (last_batch_size) {
            snp_batch.resize(last_batch_size);
            std::iota(snp_batch.begin(), snp_batch.end(), num_batches * snp_batch_size);
            // for (size_t rel_l = 0; rel_l < last_batch_size; ++rel_l) {
            //     snp_batch[rel_l] = l_ind_vec[num_batches * snp_batch_size + rel_l];
            // }
            effect_se_approx(snp_batch);
        }
    } else {
        std::vector<size_t> snp_batch(G);
        std::iota(snp_batch.begin(), snp_batch.end(), 0);
        effect_se_approx(snp_batch);
    }
}

void
MESA::effect_se() {
    q_grad2.setZero();
    q_phi_grad2.setZero();

    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        
        for (size_t l = 0; l < G; ++l)
        {
            if (genodata[i * G + l] < 9) {
                uint8_t g = genodata[i * G + l];
                MESA_NUMERIC_TYPE hkgil_norm_const = 0.;

                for (size_t k = 0; k < K; ++k)
                {
                    size_t abs_offset = (k * G + l) * (cov_num + 1);
                    
                    t_pl[k] = P_params[abs_offset];

                    for (size_t j = 0; j < cov_num; ++j) {
                        t_pl[k] += P_params[abs_offset + j + 1] * covdata[i][j];
                    }
                    
                    t_pl[k] = sigmoid(t_pl[k]);
                    
                    t_old_pl[k] = fkg(g, t_pl[k], Q[i*K+k]); 
                    hkgil_norm_const += t_old_pl[k];
                }
                
                for (size_t k = 0; k < K; ++k)
                {
                    MESA_NUMERIC_TYPE mtpl1 = t_old_pl[k] / hkgil_norm_const;
                    q_grad2[k * N + i] += mtpl1 * mtpl1 / (Q[i*K+k] * Q[i*K+k]);
                    
                    mtpl1 *= (1 - mtpl1) * ((MESA_NUMERIC_TYPE)g - 2 * t_pl[k]) / Q[i*K+k];
                    q_phi_grad2(k * N + i, l * (cov_num + 1)) = mtpl1;
                    
                    for (size_t j = 1; j <= cov_num; ++j) {
                        q_phi_grad2(k * N + i, l * (cov_num + 1) + j) = mtpl1 * covdata[i][j - 1];
                    }
                }
            } else {
                for (size_t k = 0; k < K; ++k)
                {
                    size_t abs_offset = (k * G + l) * (cov_num + 1);
                    
                    t_pl[k] = P_params[abs_offset];

                    for (size_t j = 0; j < cov_num; ++j) {
                        t_pl[k] += P_params[abs_offset + j + 1] * covdata[i][j];
                    }
                    
                    t_pl[k] = sigmoid(t_pl[k]);
                }
                
                for (size_t k = 0; k < K; ++k)
                {
                    for (uint8_t g = 0; g <= 2; ++g) {
                        MESA_NUMERIC_TYPE mtpl1 = fkg(g, t_pl[k], Q[i*K+k]);
                        q_grad2[k * N + i] += mtpl1 * mtpl1 / (Q[i*K+k] * Q[i*K+k]);
                        
                        mtpl1 *= (1 - mtpl1) * ((MESA_NUMERIC_TYPE)g - 2 * t_pl[k]) / Q[i*K+k];
                        q_phi_grad2(k * N + i, l * (cov_num + 1)) += mtpl1;
                        
                        for (size_t j = 1; j <= cov_num; ++j) {
                            q_phi_grad2(k * N + i, l * (cov_num + 1) + j) += mtpl1 * covdata[i][j - 1];
                        }
                    }
                }
            }
        }
    }

    q_grad2 = q_grad2.array().abs().array().rsqrt();
    q_phi_grad2 = q_phi_grad2.array().colwise() * q_grad2.array();

    #ifdef DEBUG_PRINT
    std::cerr << "test1" << std::endl;
    #endif
    
    size_t rel_offset = G * (cov_num + 1);
    for (size_t k = 0; k < K; ++k) {
        acc->block(0, k * rel_offset, rel_offset, rel_offset) = q_phi_grad2.block(k * N, 0, N, rel_offset).transpose() * q_phi_grad2.block(k * N, 0, N, rel_offset);
    }

    #ifdef DEBUG_PRINT
    std::cerr << "test2" << std::endl;
    #endif

    #pragma omp parallel for
    for (size_t l = 0; l < G ; ++l) {
        for (size_t i = 0; i < N; ++i) {
            if (genodata[i * G + l] < 9) {
                uint8_t g = genodata[i * G + l];
                MESA_NUMERIC_TYPE hkgil_norm_const = 0.;

                for (size_t k = 0; k < K; ++k)
                {
                    size_t abs_offset = (k * G + l) * (cov_num + 1);                    
                    t_pl[k] = P_params[abs_offset];

                    for (size_t j = 0; j < cov_num; ++j) {
                        t_pl[k] += P_params[abs_offset + j + 1] * covdata[i][j];
                    }

                    t_pl[k] = sigmoid(t_pl[k]);

                    t_old_pl[k] = fkg(g, t_pl[k], Q[i*K+k]);
                    hkgil_norm_const += t_old_pl[k];
                }

                for (size_t k = 0; k < K; ++k)
                {
                    MESA_NUMERIC_TYPE mtpl2 = t_old_pl[k] / hkgil_norm_const;
                    MESA_NUMERIC_TYPE mtpl1 = 2. * t_pl[k] * (1 - t_pl[k]) * mtpl2;
                    MESA_NUMERIC_TYPE allele_err = (MESA_NUMERIC_TYPE)g - 2. * t_pl[k];
                    mtpl2 *= (1 - mtpl2) * allele_err * allele_err;
                    
                    size_t rel_offset = G * (cov_num + 1);
                    (*acc)(l * (cov_num + 1), k * rel_offset + l * (cov_num + 1)) += mtpl2 - mtpl1;
                    for (size_t j1 = 1; j1 <= cov_num; ++j1) {
                        (*acc)(l * (cov_num + 1), k * rel_offset + l * (cov_num + 1) + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1];
                        (*acc)(l * (cov_num + 1) + j1, k * rel_offset + l * (cov_num + 1)) += (mtpl2 - mtpl1) * covdata[i][j1-1];
                        (*acc)(l * (cov_num + 1) + j1, k * rel_offset + l * (cov_num + 1) + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j1-1];
                        
                        for (size_t j2 = j1 + 1; j2 <= cov_num; ++j2) {
                            (*acc)(l * (cov_num + 1) + j1, k * rel_offset + l * (cov_num + 1) + j2) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j2-1];
                            (*acc)(l * (cov_num + 1) + j2, k * rel_offset + l * (cov_num + 1) + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j2-1];
                        }
                    }
                }
            } else {
                for (size_t k = 0; k < K; ++k)
                {
                    size_t abs_offset = (k * G + l) * (cov_num + 1);                    
                    t_pl[k] = P_params[abs_offset];

                    for (size_t j = 0; j < cov_num; ++j) {
                        t_pl[k] += P_params[abs_offset + j + 1] * covdata[i][j];
                    }

                    t_pl[k] = sigmoid(t_pl[k]);
                }

                for (size_t k = 0; k < K; ++k)
                {
                    for (uint8_t g = 0; g <= 2; ++g) {
                        MESA_NUMERIC_TYPE mtpl2 = fkg(g, t_pl[k], Q[i*K+k]);
                        MESA_NUMERIC_TYPE mtpl1 = 2. * t_pl[k] * (1 - t_pl[k]) * mtpl2;
                        MESA_NUMERIC_TYPE allele_err = (MESA_NUMERIC_TYPE)g - 2. * t_pl[k];
                        mtpl2 *= (1 - mtpl2) * allele_err * allele_err;
                        
                        size_t rel_offset = G * (cov_num + 1);
                        (*acc)(l * (cov_num + 1), k * rel_offset + l * (cov_num + 1)) += mtpl2 - mtpl1;
                        for (size_t j1 = 1; j1 <= cov_num; ++j1) {
                            (*acc)(l * (cov_num + 1), k * rel_offset + l * (cov_num + 1) + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1];
                            (*acc)(l * (cov_num + 1) + j1, k * rel_offset + l * (cov_num + 1)) += (mtpl2 - mtpl1) * covdata[i][j1-1];
                            (*acc)(l * (cov_num + 1) + j1, k * rel_offset + l * (cov_num + 1) + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j1-1];

                            for (size_t j2 = j1 + 1; j2 <= cov_num; ++j2) {
                                (*acc)(l * (cov_num + 1) + j1, k * rel_offset + l * (cov_num + 1) + j2) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j2-1];
                                (*acc)(l * (cov_num + 1) + j2, k * rel_offset + l * (cov_num + 1) + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j2-1];
                            }
                        }
                    }
                }
            }
        }
    }
    
    #ifdef DEBUG_PRINT
    std::cerr << "test3" << std::endl;
    #endif

    rel_offset = G * (cov_num + 1);
    #pragma omp parallel for
    for (size_t k = 0; k < K; ++k) {
        // MESA_MATRIX_TYPE I = MESA_MATRIX_TYPE::Identity(rel_offset, rel_offset);
        // acc->block(0, k * rel_offset, rel_offset, rel_offset) = (acc->block(0, k * rel_offset, rel_offset, rel_offset)).llt().solve(I);
        acc->block(0, k * rel_offset, rel_offset, rel_offset) = (-acc->block(0, k * rel_offset, rel_offset, rel_offset)).inverse();
        for (size_t l = 0; l < G ; ++l) {
            for (size_t j = 1; j <= cov_num; ++j) {
                effect_se_ptr[(k * G + l) * cov_num + j - 1] = std::sqrt(abs((*acc)(l * (cov_num + 1) + j, k * rel_offset + l * (cov_num + 1) + j)));
            }
        }
    }

    #ifdef DEBUG_PRINT
    std::cerr << "test4" << std::endl;
    #endif
}

void
MESA::effect_se_approx(const std::vector<size_t> &snp_batch) {
    size_t _snp_batch_size = snp_batch.size();
    q_grad2.setZero();
    q_phi_grad2.setZero();

    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        
        for (size_t rel_l = 0; rel_l < _snp_batch_size ; ++rel_l)
        {
            if (genodata[i * G + snp_batch[rel_l]] < 9) {
                uint8_t g = genodata[i * G + snp_batch[rel_l]];
                MESA_NUMERIC_TYPE hkgil_norm_const = 0.;

                for (size_t k = 0; k < K; ++k)
                {
                    size_t abs_offset = (k * G + snp_batch[rel_l]) * (cov_num + 1);
                    
                    t_pl[k] = P_params[abs_offset];

                    for (size_t j = 0; j < cov_num; ++j) {
                        t_pl[k] += P_params[abs_offset + j + 1] * covdata[i][j];
                    }
                    
                    t_pl[k] = sigmoid(t_pl[k]);
                    
                    // t_old_pl[k] = fkg(g, t_pl[k], Q[i*K+k]);
                    t_old_pl[k] = fkg(g, t_pl[k]);
                    // t_old_pl[k] = fkg(g, t_pl[k], 1./K);
                    hkgil_norm_const += t_old_pl[k] * Q[i*K+k];
                }
                
                for (size_t k = 0; k < K; ++k)
                {
                    MESA_NUMERIC_TYPE mtpl1 = t_old_pl[k] / hkgil_norm_const;
                    // q_grad2[k * N + i] -= mtpl1 * mtpl1 / (Q[i*K+k] * Q[i*K+k]);
                    q_grad2[k * N + i] += mtpl1 * mtpl1;
                    // q_grad2[k * N + i] -= mtpl1 * mtpl1 * K * K;
                    
                    mtpl1 *= (1 - mtpl1 * Q[i * K + k]) * ((MESA_NUMERIC_TYPE)g - 2 * t_pl[k]);
                    // mtpl1 *= (1 - mtpl1) * ((MESA_NUMERIC_TYPE)g - 2 * t_pl[k]) / Q[i*K+k];
                    // mtpl1 *= (1 - mtpl1) * ((MESA_NUMERIC_TYPE)g - 2 * t_pl[k]) * K;
                    q_phi_grad2(k * N + i, rel_l * (cov_num + 1)) = mtpl1;
                    
                    for (size_t j = 1; j <= cov_num; ++j) {
                        q_phi_grad2(k * N + i, rel_l * (cov_num + 1) + j) = mtpl1 * covdata[i][j - 1];
                    }
                }
            } else {
                for (size_t k = 0; k < K; ++k)
                {
                    size_t abs_offset = (k * G + snp_batch[rel_l]) * (cov_num + 1);
                    
                    t_pl[k] = P_params[abs_offset];

                    for (size_t j = 0; j < cov_num; ++j) {
                        t_pl[k] += P_params[abs_offset + j + 1] * covdata[i][j];
                    }
                    
                    t_pl[k] = sigmoid(t_pl[k]);
                }
                
                for (size_t k = 0; k < K; ++k)
                {
                    for (uint8_t g = 0; g <= 2; ++g) {
                        MESA_NUMERIC_TYPE mtpl1 = fkg(g, t_pl[k]);
                        // MESA_NUMERIC_TYPE mtpl1 = fkg(g, t_pl[k], Q[i*K+k]);
                        q_grad2[k * N + i] += mtpl1 * mtpl1;
                        // q_grad2[k * N + i] -= mtpl1 * mtpl1 / (Q[i*K+k] * Q[i*K+k]);
                        // q_grad2[k * N + i] -= mtpl1 * mtpl1 * K * K;
                        
                        mtpl1 *= (1 - mtpl1 * Q[i * K + k]) * ((MESA_NUMERIC_TYPE)g - 2 * t_pl[k]);
                        // mtpl1 *= (1 - mtpl1) * ((MESA_NUMERIC_TYPE)g - 2 * t_pl[k]) / Q[i*K+k];
                        // mtpl1 *= (1 - mtpl1) * ((MESA_NUMERIC_TYPE)g - 2 * t_pl[k]) * K;
                        q_phi_grad2(k * N + i, rel_l * (cov_num + 1)) += mtpl1;
                        
                        for (size_t j = 1; j <= cov_num; ++j) {
                            q_phi_grad2(k * N + i, rel_l * (cov_num + 1) + j) += mtpl1 * covdata[i][j - 1];
                        }
                    }
                }
            }
        }
    }

    q_grad2 = q_grad2.array().abs().array().rsqrt();
    q_phi_grad2 = q_phi_grad2.array().colwise() * q_grad2.array();

    #ifdef DEBUG_PRINT
    std::cerr << "test1" << std::endl;
    #endif
    
    #ifdef BLOCK_SE_APPROX
    size_t rel_offset = _snp_batch_size * (cov_num + 1);
    acc->setZero();
    for (size_t k = 0; k < K; ++k) {
        acc->block(0, k * rel_offset, rel_offset, rel_offset) = q_phi_grad2.block(k * N, 0, N, rel_offset).transpose() * q_phi_grad2.block(k * N, 0, N, rel_offset);
    }
    #endif

    // #ifdef SNPWISE_SE_NEUMANN_APPROX
    // MESA_MATRIX_TYPE block_identity = MESA_MATRIX_TYPE::Identity(cov_num + 1, cov_num + 1);
    // #endif
    
    #ifdef DEBUG_PRINT
    std::cerr << "test2" << std::endl;
    #endif

    #pragma omp parallel for
    for (size_t rel_l = 0; rel_l < _snp_batch_size ; ++rel_l)
    {
        #ifdef SNPWISE_SE_APPROX
        for (size_t k = 0; k < K; ++k) {
            size_t rel_offset = rel_l * (cov_num + 1);
            acc->block(0, k * (cov_num + 1), cov_num + 1, cov_num + 1) = q_phi_grad2.block(k * N, rel_offset, N, cov_num + 1).transpose() * q_phi_grad2.block(k * N, rel_offset, N, cov_num + 1);       
        }
        #endif
        
        #ifdef SNPWISE_SE_NEUMANN_APPROX
        acc->setZero();
        for (size_t k = 0; k < K; ++k) {
            size_t rel_offset = rel_l * (cov_num + 1);
            M->block(0, k * (cov_num + 1), cov_num + 1, cov_num + 1) = q_phi_grad2.block(k * N, rel_offset, N, cov_num + 1).transpose() * q_phi_grad2.block(k * N, rel_offset, N, cov_num + 1);       
            R->block(0, k * (cov_num + 1), cov_num + 1, cov_num + 1) = MESA_MATRIX_TYPE::Identity(cov_num + 1, cov_num + 1);
        }
        #endif
        
        for (size_t i = 0; i < N; ++i) {
            if (genodata[i * G + snp_batch[rel_l]] < 9) {
                uint8_t g = genodata[i * G + snp_batch[rel_l]];
                MESA_NUMERIC_TYPE hkgil_norm_const = 0.;

                for (size_t k = 0; k < K; ++k)
                {
                    size_t abs_offset = (k * G + snp_batch[rel_l]) * (cov_num + 1);                    
                    t_pl[k] = P_params[abs_offset];

                    for (size_t j = 0; j < cov_num; ++j) {
                        t_pl[k] += P_params[abs_offset + j + 1] * covdata[i][j];
                    }

                    t_pl[k] = sigmoid(t_pl[k]);

                    // t_old_pl[k] = fkg(g, t_pl[k]);
                    t_old_pl[k] = fkg(g, t_pl[k], Q[i*K+k]);
                    // t_old_pl[k] = fkg(g, t_pl[k], 1./K);
                    hkgil_norm_const += t_old_pl[k];
                }

                for (size_t k = 0; k < K; ++k)
                {
                    t_old_pl[k] /= hkgil_norm_const;
                    MESA_NUMERIC_TYPE mtpl1 = 2. * t_pl[k] * (1 - t_pl[k]) * t_old_pl[k];
                    MESA_NUMERIC_TYPE allele_err = (MESA_NUMERIC_TYPE)g - 2. * t_pl[k];
                    MESA_NUMERIC_TYPE mtpl2 = t_old_pl[k];
                    mtpl2 *= (1 - mtpl2) * allele_err * allele_err;
                    
                    #ifdef BLOCK_SE_APPROX
                    size_t rel_offset = _snp_batch_size * (cov_num + 1);
                    (*acc)(rel_l * (cov_num + 1), k * rel_offset + rel_l * (cov_num + 1)) += mtpl2 - mtpl1;
                    for (size_t j1 = 1; j1 <= cov_num; ++j1) {
                        (*acc)(rel_l * (cov_num + 1), k * rel_offset + rel_l * (cov_num + 1) + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1];
                        (*acc)(rel_l * (cov_num + 1) + j1, k * rel_offset + rel_l * (cov_num + 1)) += (mtpl2 - mtpl1) * covdata[i][j1-1];
                        (*acc)(rel_l * (cov_num + 1) + j1, k * rel_offset + rel_l * (cov_num + 1) + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j1-1];
                        
                        for (size_t j2 = j1 + 1; j2 <= cov_num; ++j2) {
                            (*acc)(rel_l * (cov_num + 1) + j1, k * rel_offset + rel_l * (cov_num + 1) + j2) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j2-1];
                            (*acc)(rel_l * (cov_num + 1) + j2, k * rel_offset + rel_l * (cov_num + 1) + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j2-1];
                        }
                    }
                    #endif

                    #if defined(SNPWISE_SE_APPROX) || defined(SNPWISE_SE_NEUMANN_APPROX)
                    size_t rel_offset = k * (cov_num + 1);
                    (*acc)(0, rel_offset) += mtpl2 - mtpl1;
                    for (size_t j1 = 1; j1 <= cov_num; ++j1) {
                        (*acc)(0, rel_offset + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1];
                        (*acc)(j1, rel_offset) += (mtpl2 - mtpl1) * covdata[i][j1-1];
                        (*acc)(j1, rel_offset + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j1-1];

                        for (size_t j2 = j1 + 1; j2 <= cov_num; ++j2) {
                            (*acc)(j1, rel_offset + j2) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j2-1];
                            (*acc)(j2, rel_offset + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j2-1];
                        }
                    }
                    #endif
                }
            } else {
                for (size_t k = 0; k < K; ++k)
                {
                    size_t abs_offset = (k * G + snp_batch[rel_l]) * (cov_num + 1);                    
                    t_pl[k] = P_params[abs_offset];

                    for (size_t j = 0; j < cov_num; ++j) {
                        t_pl[k] += P_params[abs_offset + j + 1] * covdata[i][j];
                    }

                    t_pl[k] = sigmoid(t_pl[k]);
                }

                for (size_t k = 0; k < K; ++k)
                {
                    for (uint8_t g = 0; g <= 2; ++g) {
                        // MESA_NUMERIC_TYPE t_old_pl_tmp = fkg(g, t_pl[k]);
                        MESA_NUMERIC_TYPE mtpl2 = fkg(g, t_pl[k], Q[i*K+k]);
                        // MESA_NUMERIC_TYPE t_old_pl_tmp = fkg(g, t_pl[k], 1./K);
                        MESA_NUMERIC_TYPE mtpl1 = 2. * t_pl[k] * (1 - t_pl[k]) * mtpl2;
                        MESA_NUMERIC_TYPE allele_err = (MESA_NUMERIC_TYPE)g - 2. * t_pl[k];
                        mtpl2 *= (1 - mtpl2) * allele_err * allele_err;
                        
                        #ifdef BLOCK_SE_APPROX
                        size_t rel_offset = _snp_batch_size * (cov_num + 1);
                        (*acc)(rel_l * (cov_num + 1), k * rel_offset + rel_l * (cov_num + 1)) += mtpl2 - mtpl1;
                        for (size_t j1 = 1; j1 <= cov_num; ++j1) {
                            (*acc)(rel_l * (cov_num + 1), k * rel_offset + rel_l * (cov_num + 1) + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1];
                            (*acc)(rel_l * (cov_num + 1) + j1, k * rel_offset + rel_l * (cov_num + 1)) += (mtpl2 - mtpl1) * covdata[i][j1-1];
                            (*acc)(rel_l * (cov_num + 1) + j1, k * rel_offset + rel_l * (cov_num + 1) + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j1-1];

                            for (size_t j2 = j1 + 1; j2 <= cov_num; ++j2) {
                                (*acc)(rel_l * (cov_num + 1) + j1, k * rel_offset + rel_l * (cov_num + 1) + j2) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j2-1];
                                (*acc)(rel_l * (cov_num + 1) + j2, k * rel_offset + rel_l * (cov_num + 1) + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j2-1];
                            }
                        }
                        #endif
                        
                        #if defined(SNPWISE_SE_APPROX) || defined(SNPWISE_SE_NEUMANN_APPROX)
                        size_t rel_offset = k * (cov_num + 1);
                        (*acc)(0, rel_offset) += mtpl2 - mtpl1;
                        for (size_t j1 = 1; j1 <= cov_num; ++j1) {
                            (*acc)(0, rel_offset + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1];
                            (*acc)(j1, rel_offset) += (mtpl2 - mtpl1) * covdata[i][j1-1];
                            (*acc)(j1, rel_offset + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j1-1];

                            for (size_t j2 = j1 + 1; j2 <= cov_num; ++j2) {
                                (*acc)(j1, rel_offset + j2) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j2-1];
                                (*acc)(j2, rel_offset + j1) += (mtpl2 - mtpl1) * covdata[i][j1-1] * covdata[i][j2-1];
                            }
                        }
                        #endif
                    }
                }
            }
        }
        #ifdef SNPWISE_SE_APPROX
        for (size_t k = 0; k < K; ++k) {
            size_t rel_offset = k * (cov_num + 1);
            acc->block(0, rel_offset, cov_num + 1, cov_num + 1) = (acc->block(0, rel_offset, cov_num + 1, cov_num + 1)).inverse();
            for (size_t j = 1; j <= cov_num; ++j) {
                // effect_se_ptr[(k * G + snp_batch[rel_l]) * cov_num + j - 1] = std::sqrt(-(*acc)(j, rel_offset + j));
                effect_se_ptr[(k * G + snp_batch[rel_l]) * cov_num + j - 1] = std::sqrt(abs((*acc)(j, rel_offset + j)));
            }
        }
        #endif

        #ifdef SNPWISE_SE_NEUMANN_APPROX
        for (size_t k = 0; k < K; ++k) {
            size_t rel_offset = k * (cov_num + 1);
            acc->block(0, rel_offset, cov_num + 1, cov_num + 1) = (acc->block(0, rel_offset, cov_num + 1, cov_num + 1)).inverse();
            M->block(0, rel_offset, cov_num + 1, cov_num + 1) = M->block(0, rel_offset, cov_num + 1, cov_num + 1) * acc->block(0, rel_offset, cov_num + 1, cov_num + 1);

            for (size_t ii = 0; ii < NEUMANN_APPROX_NUM_ITER; ++ii) {
                R->block(0, rel_offset, cov_num + 1, cov_num + 1) = MESA_MATRIX_TYPE::Identity(cov_num + 1, cov_num + 1) + M->block(0, rel_offset, cov_num + 1, cov_num + 1) * R->block(0, rel_offset, cov_num + 1, cov_num + 1);
            }

            acc->block(0, rel_offset, cov_num + 1, cov_num + 1) = acc->block(0, rel_offset, cov_num + 1, cov_num + 1) * R->block(0, rel_offset, cov_num + 1, cov_num + 1);
            for (size_t j = 1; j <= cov_num; ++j) {
                effect_se_ptr[(k * G + snp_batch[rel_l]) * cov_num + j - 1] = std::sqrt(abs((*acc)(j, rel_offset + j)));
            }
        }
        #endif
    }
    
    #ifdef DEBUG_PRINT
    std::cerr << "test3" << std::endl;
    #endif
    #ifdef BLOCK_SE_APPROX
    rel_offset = _snp_batch_size * (cov_num + 1);
    #pragma omp parallel for
    for (size_t k = 0; k < K; ++k) {
        // MESA_MATRIX_TYPE I = MESA_MATRIX_TYPE::Identity(rel_offset, rel_offset);
        // acc->block(0, k * rel_offset, rel_offset, rel_offset) = (acc->block(0, k * rel_offset, rel_offset, rel_offset)).llt().solve(I);
        acc->block(0, k * rel_offset, rel_offset, rel_offset) = acc->block(0, k * rel_offset, rel_offset, rel_offset).inverse();
        // acc->block(0, k * rel_offset, rel_offset, rel_offset) -= acc->block(0, k * rel_offset, rel_offset, rel_offset) * q_phi_grad2.block(k * N, 0, N, rel_offset).transpose() * q_phi_grad2.block(k * N, 0, N, rel_offset) * acc->block(0, k * rel_offset, rel_offset, rel_offset);
        for (size_t rel_l = 0; rel_l < _snp_batch_size ; ++rel_l) {
            for (size_t j = 1; j <= cov_num; ++j) {
                effect_se_ptr[(k * G + snp_batch[rel_l]) * cov_num + j - 1] = std::sqrt(abs((*acc)(rel_l * (cov_num + 1) + j, k * rel_offset + rel_l * (cov_num + 1) + j)));
            }
        }
    }
    #endif
    #ifdef DEBUG_PRINT
    std::cerr << "test4" << std::endl;
    #endif
}

void
MESA::save_effect_se(const std::string output_path) {
    std::string effect_se_path;
    size_t offset = 0;
    effect_se_path = output_path + "_effect_se.tsv";
    
    std::ofstream ofile(effect_se_path, std::ios::out);
    
    for (size_t l = 0; l < G; ++l)
    {
        for (size_t k = 0; k < K; ++k)
        {
            offset = (k * G + l) * cov_num;
            for (size_t j = 0; j < cov_num; ++j)
            {
                ofile << std::fixed << std::setprecision(SAVED_VALUES_PREC) << effect_se_ptr[offset + j];
                ofile << ((k < K - 1 || j < cov_num - 1) ? "\t" : "\n");
            }
        }
    }
    ofile.close();
}

void
show_min_memory_alloc(const Config& c, size_t cov_num) {
    size_t N_ = c.get_N(), K_ = c.get_K(), G_ = c.get_G(), R_ = cov_num;
    size_t B_ = std::min(G_, (size_t)c.get_batch_size());
    size_t P_;
    #pragma omp parallel
    {
        #pragma omp single
        P_ = omp_get_num_threads();
    }
    size_t min_usage = (
        N_*G_*1 // genodata
        + N_*R_*ENTRY_SIZE // covdata
        + K_*P_*ENTRY_SIZE // t_pl
        + K_*P_*ENTRY_SIZE // t_old_pl
        + K_*ENTRY_SIZE // pl
        + K_*ENTRY_SIZE // old_pl
        + K_*G_*(R_+1)*ENTRY_SIZE // P_params
        + K_*G_*R_*ENTRY_SIZE // effect_se_ptr
        + N_*K_*ENTRY_SIZE // Q
        + N_*K_*ENTRY_SIZE // Q_buffer
        + G_*ENTRY_SIZE // l_ind_vec
        + B_*1 // not_converged_snps
        + B_*1 // globally_not_converged_snps
        + K_*B_*(R_+1)*ENTRY_SIZE // buffer
        + K_*B_*(R_+1)*ENTRY_SIZE // result
        + N_*K_*ENTRY_SIZE // q_grad2
        #ifdef SNPWISE_SE_APPROX
        + N_*K_*B_*(R_+1)*ENTRY_SIZE // q_phi_grad2
        + (R_+1)*(R_+1)*K_*P_*ENTRY_SIZE // acc
        #endif
        #ifdef SNPWISE_SE_APPROX
        + N_*K_*B_*(R_+1)*ENTRY_SIZE // q_phi_grad2
        + (R_+1)*(R_+1)*K_*P_*ENTRY_SIZE // acc
        + (R_+1)*(R_+1)*K_*P_*ENTRY_SIZE // R
        + (R_+1)*(R_+1)*K_*P_*ENTRY_SIZE // M
        #endif
        #ifdef BLOCK_SE_APPROX
        + N_*K_*B_*(R_+1)*ENTRY_SIZE // q_phi_grad2
        + B_*B_*(R_+1)*(R_+1)*K_*ENTRY_SIZE // acc
        #endif
        #ifdef FULL_SE
        + N_*K_*G_*(R_+1)*ENTRY_SIZE // q_phi_grad2
        + G_*G_*(R_+1)*(R_+1)*K_*ENTRY_SIZE // acc
        #endif
        + B_*ENTRY_SIZE // snp_batch
        + K_*G_*(R_+1)*ENTRY_SIZE // old_P_params
        + K_*G_*ENTRY_SIZE // P_params_norm_const
        + K_*G_*(R_+1)*ENTRY_SIZE // new_P_params
        + N_*K_*ENTRY_SIZE // old_Q
        + N_*K_*ENTRY_SIZE // new_Q
        + N_*B_*K_*ENTRY_SIZE // hkgil_by_il
        + K_*B_*(R_+1)*ENTRY_SIZE // grad1
        + K_*B_*(R_+1)*(R_+1)*ENTRY_SIZE // grad2
    );

    const size_t KB_THRESHOLD = 1024;
    const size_t MB_THRESHOLD = KB_THRESHOLD * 1024;
    const size_t GB_THRESHOLD = MB_THRESHOLD * 1024;

    std::cout << "Suggested minimum memory to allocate for this dataset: ";
    if (min_usage >= GB_THRESHOLD) {
        std::cout << std::fixed << std::setprecision(2) << (MESA_NUMERIC_TYPE)min_usage/GB_THRESHOLD << " GB";
    } else if (min_usage >= MB_THRESHOLD) {
        std::cout << std::fixed << std::setprecision(2) << (MESA_NUMERIC_TYPE)min_usage/MB_THRESHOLD << " MB";
    } else if (min_usage >= KB_THRESHOLD) {
        std::cout << std::fixed << std::setprecision(2) << (MESA_NUMERIC_TYPE)min_usage/KB_THRESHOLD << " KB";
    } else {
        std::cout << min_usage << " Bytes";
    }
    std::cout << std::endl;
}

int main(int argc, const char **argv)
{
    if (argc <= 1) {
        Config::show_available_options();
        return 0;
    }

    #ifdef FAST_SIGMOID
    init_sigmoid_tbl();
    #endif

    t0_tp = std::chrono::high_resolution_clock::now();
    t0 = std::chrono::high_resolution_clock::to_time_t(t0_tp);
    std::cout << "Time now: " << std::put_time(std::localtime(&t0), "%F %T") << std::endl;

    Config c = Config();
    c.parse_config(argc, argv);
    CovDataReader cdr = CovDataReader().open(c);

    show_min_memory_alloc(c, cdr.shape[1]);

    GenoDataReader gdr;
    std::string const geno_input_path = c.get_geno_input_path();

    if ( geno_input_path.rfind(".tsv") != std::string::npos ) {
        gdr = GenoDataReader().open(c);
        // geno_file_type = GenoFileT::tsv;
    } else if ( geno_input_path.rfind(".ttsv") != std::string::npos ) {
        gdr = GenoDataReader().open_t(c);
        // geno_file_type = GenoFileT::ttsv;
    } else if ( geno_input_path.rfind(".bed") != std::string::npos ) {
        gdr = GenoDataReader().open_bed(c);
        // geno_file_type = GenoFileT::bed;
    } else {
        throw InvalidGenoFileError();
    }

    std::cout << gdr.shape[0] << ", " << gdr.shape[1] << "\n";
    std::cout << gdr.get_input_path() << "\n";
    std::cout << cdr.shape[0] << ", " << cdr.shape[1] << "\n";
    std::cout << cdr.get_input_path() << "\n";

    MESA ems(&gdr, &cdr, &c);

    if (c._continue) {
        ResultDataReader rdr = ResultDataReader(c, cdr.shape[1]);
        ems.load_est(rdr);
        std::cout << "Loaded old parameter estimates." << std::endl;
    }
    // ems.init_metrics_file(c.get_output_path());
    // std::cout << "Initialized metrics file." << std::endl;

    if (!c.se_only) {
        if (c._pretrain) ems.pretrain();
        ems.train();
    } 
    
    if (ems.get_cov_num() > 0) {
        #ifdef FULL_SE
        ems.effect_se();
        #else
        ems.effect_se_full_approx();
        #endif
        ems.save_effect_se(c.get_output_path());
    }

    std::cout << "Finished training..." << "\n";
    t2_tp = std::chrono::high_resolution_clock::now();
    t2 = std::chrono::high_resolution_clock::to_time_t(t2_tp);
    std::cout << "Training finishes at: " << std::put_time(std::localtime(&t2), "%F %T") << ", "
              << "Time elapsed: " << std::chrono::duration_cast<std::chrono::hours>(t2_tp - t0_tp).count() << " h "
              << std::chrono::duration_cast<std::chrono::minutes>(t2_tp - t0_tp).count() % 60 << " mins\n" << std::endl;

    gdr.close();
    cdr.close();

    #if defined(BLOCK_SE_APPROX) || defined(FULL_SE)
    delete acc;
    #endif

    #ifdef SNPWISE_SE_APPROX
    #pragma omp parallel
    {
        delete acc;
        delete[] t_pl;
        delete[] t_old_pl;
    }
    #endif
    
    #ifdef SNPWISE_SE_NEUMANN_APPROX
    #pragma omp parallel
    {
        delete acc;
        delete R;
        delete M;
        delete[] t_pl;
        delete[] t_old_pl;
    }
    #endif
    return 0;
}
