#ifndef __MESA_H
#define __MESA_H

#include "common.h"
#include "utils.h"
#include <vector>
// #include "/home/m2ng/.local/include/eigen-3.3.7/Eigen/Core"
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Cholesky>


#ifdef PRECISION_DOUBLE
#define MESA_MATRIX_TYPE Eigen::MatrixXd
#define MESA_VECTOR_TYPE Eigen::VectorXd
#endif

#ifdef PRECISION_FLOAT
#define MESA_MATRIX_TYPE Eigen::MatrixXf
#define MESA_VECTOR_TYPE Eigen::VectorXf
#endif


inline MESA_NUMERIC_TYPE fkg(uint8_t g, MESA_NUMERIC_TYPE pkl);
inline MESA_NUMERIC_TYPE fkg(uint8_t g, MESA_NUMERIC_TYPE pkl, MESA_NUMERIC_TYPE qik);

class MESA
{
protected:
    size_t N = 0, K = 0, G = 0;
    MESA_NUMERIC_TYPE sqrtN, inv_sqrtN, inv_N, sqrtG, lg2G;
    size_t snp_batch_size;
    Config *config;
    GenoData genodata;
    CovData covdata;
    MESA_VECTOR_TYPE P_params;
    size_t cov_num;
    MESA_NUMERIC_TYPE *Q, *Q_buffer;
    std::vector<size_t> l_ind_vec;
    std::vector<bool> not_converged_snps;
    std::vector<bool> not_converged_subjects;
    std::vector<bool> globally_not_converged_snps;
    MESA_NUMERIC_TYPE* effect_se_ptr = nullptr;
    MESA_NUMERIC_TYPE metrics[5 * SAVE_INTERVAL];

    void init_params();
    void update(const std::vector<size_t>&, size_t, MESA_VECTOR_TYPE &, MESA_MATRIX_TYPE &, MESA_NUMERIC_TYPE **);
    
    void compute_grad(const MESA_VECTOR_TYPE &x, const std::vector<size_t>&, MESA_VECTOR_TYPE &grad1, MESA_MATRIX_TYPE &grad2);
    void block_newton_iteration(MESA_VECTOR_TYPE &x, const std::vector<size_t>&, const MESA_VECTOR_TYPE &grad1, const MESA_MATRIX_TYPE &grad2);
    void snp_wise_newton_update(MESA_VECTOR_TYPE &x, const std::vector<size_t>&, MESA_VECTOR_TYPE &grad1, MESA_MATRIX_TYPE &grad2);
    MESA_NUMERIC_TYPE subset_obs_logl(const MESA_VECTOR_TYPE &, const std::vector<size_t>&);
public:
    MESA(GenoDataReader *gdr, CovDataReader *cdr, Config *config);
    void load_est(ResultDataReader&);
    void train();
    void pretrain();

    inline size_t get_N() { return N; };
    inline size_t get_K() { return K; };
    inline size_t get_G() { return G; };
    inline const MESA_VECTOR_TYPE &get_P_params() { return P_params; };
    inline const MESA_NUMERIC_TYPE *get_Q() { return Q; };
    inline GenoData get_genodata() { return genodata; }
    inline CovData get_covdata() { return covdata; }
    inline size_t get_cov_num() { return cov_num; }

    MESA_NUMERIC_TYPE obs_logl(const MESA_VECTOR_TYPE &);
    MESA_NUMERIC_TYPE obs_logl_2(const MESA_VECTOR_TYPE &);

    inline void print_Q_for_I(size_t);
    inline void print_P_for_G(size_t);

    void save_params(const std::string);
    // void save_BIC(const std::string);
    // void init_metrics_file(const std::string);
    // void save_metrics(const std::string, size_t);

    void save_effect_se(const std::string);

    void effect_se_full_approx();
    void effect_se_approx(const std::vector<size_t>&);
    void effect_se();

    ~MESA();
};

#endif