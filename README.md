# MESA
MESA is a method to call ethnicity-specific associations for biobank-scale multi-ethnic GWAS.

 MESA is only available on Linux.

## Requirements
1. A working Linux environment
2. Intel CPU that supports OpenMP and AVX (Tested on Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz)
3. GCC >= 9 and build-essential
4. Eigen >= 3.3.7 from [here](https://eigen.tuxfamily.org)

## Installation
1. Specify variables `BUILD_DIR`, `SRC_DIR`, `EIGEN_DIR` in `makefile`
2. Specify the user options in `common.h` if necessary.
3. Run `make` at the project root to build MESA

## How-to-use
1. Export the environment variable `OMP_NUM_THREADS` before using MESA. e.g.
    ```bash
    export OMP_NUM_THREADS=4
    ```
2. Run `./build/mesa` to see available options
3. Follow the instructions [here](./example/README.md) to see how you may apply MESA to the data set you would like to analyze

## Help
```
Usage: mesa -i INPUT [--cov COV_FILE] -N NUM_INDIVIDUALS -K NUM_ANCESTRIES -G NUM_SNPS -o OUTPUT_PREFIX [OPTION]...

Required arguments:
   -i, --in INPUT              Path of genotype matrix file (String). 
                               Accepted file type: tsv, ttsv, bed
                               e.g. -i /home/User/genotype.bed
   -N NUM_INDIVIDUALS          Number of individuals to be tested
                               (Positive integer)
   -K NUM_ANCESTRIES           Number of ancestries to be fitted
                               (Positive integer)
   -G NUM_SNPS                 Number of SNPs to be tested (Positive
                               integer)
   -o, --out OUTPUT_PREFIX     Prefix of path of output file
                               (String). For example, option
                               '-o /home/User/output' will produce
                               files such as '/home/User/output_p.tsv'Optional arguments:
   --cov COV_FILE              Path of phenotype matrix file (String)
   --batch-size                Number of subsampled SNPs (Positive
                               integer, 1000
   --newton-iter X             Maximum number of Newton's steps = X
                               (Positive integer, default: 10)
   --em-iter X                 Maximum number of E-steps and M-steps = X
                               (Positive integer, default: 30)
   --epochs X                  Number of passes through whole data = X
                               set (Positive integer, default: 1)
   --continue                  Whether to use existing estimates
                               extracted from OUTPUT_PREFIX_p.tsv, etc
   --no-pretrain               Start training with no warm-up
   --se-only                   Calculate standard errors using
                               existing estimates extracted from
                               OUTPUT_PREFIX_p.tsv,
                               OUTPUT_PREFIX_q.tsv and 
                               OUTPUT_PREFIX_effect.tsv
```

## Output
- `*_p.tsv`: a G-by-K matrix of estimated baseline allele frequency intercept terms. The (g, k)-th entry corresponds to <img src="https://render.githubusercontent.com/render/math?math=\alpha_{gk}"> in the literature
- `*_q.tsv`: a N-by-K matrix of estimated ancestry proportions. The (i, k)-th entry corresponds to <img src="https://render.githubusercontent.com/render/math?math=q_{ki}"> in the literature
- `*_effect.tsv`: a G-by-(K * cov_num) matrix of estimated effect sizes, where 'cov_num' is the number of phenotypes tested. The (g, (k-1)*cov_num + r)-th entry corresponds to the estimate of <img src="https://render.githubusercontent.com/render/math?math=\gamma_{gkr}"> in the literature
- `*_effect_se.tsv`: a G-by-(K * cov_num) matrix of standard errors of estimated effect sizes, where 'cov_num' is the number of phenotypes tested. The (g, (k-1)*cov_num + r)-th entry corresponds to the standard error estimate of <img src="https://render.githubusercontent.com/render/math?math=\gamma_{gkr}"> in the literature
