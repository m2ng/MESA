# Example data set

## Description
This is an example data set of N=2000 individuals genotyped at G=4000 SNPs. The true number of ancestries K is equal to 3. The accuracy of the estimates should be lower because the number of individuals is small.

## Prerequisites
1. Build MESA from source at the project root.

## Situations
### Consider phenotypes
1. Suppose the number of threads you want to allocate for MESA is 4. Run
    ```bash
    export OMP_NUM_THREADS=4
    ./build/mesa \
        -i ./example/N2000_K3_G4000_twisted_f6bfef.bed \
        -N 2000 \
        -K 3 \
        -G 4000 \
        --cov ./example/N2000_K3_G4000_twisted_f6bfef_R2_cov.tsv \
        -o ./example/N2000_K3_G4000_twisted_f6bfef

    ```
    to fit a K=3 model to the example data set.

2. If MESA runs successfully, it will produce four files
    ```
    ./example/N2000_K3_G4000_twisted_f6bfef_p.tsv
    ./example/N2000_K3_G4000_twisted_f6bfef_q.tsv
    ./example/N2000_K3_G4000_twisted_f6bfef_effect.tsv
    ./example/N2000_K3_G4000_twisted_f6bfef_effect_se.tsv
    ```

### Do not consider phenotypes
1. Suppose the number of threads you want to allocate for MESA is 4. Run
    ```bash
    export OMP_NUM_THREADS=4
    ./build/mesa \
        -i ./example/N2000_K3_G4000_twisted_f6bfef.bed \
        -N 2000 \
        -K 3 \
        -G 4000 \
        -o ./example/N2000_K3_G4000_twisted_f6bfef

    ```
    to fit a K=3 model to the example data set.

2. If MESA runs successfully, it will produce two files
    ```
    ./example/N2000_K3_G4000_twisted_f6bfef_p.tsv
    ./example/N2000_K3_G4000_twisted_f6bfef_q.tsv
    ```