#ifndef __RNG_H
#define __RNG_H

#include <random>
#include <chrono>
#include <iterator>
#include <algorithm>


class RNG {
    std::mt19937 mte;
    unsigned int mte_max;

    public:
        RNG() { 
            mte = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
            mte_max = mte.max();
        }
        RNG(size_t seed) {
            mte = std::mt19937(seed);
            mte_max = mte.max();
        }
        int rand(int l, int u) {
            return mte() % (u - l + 1) + l;
        }
        float rand() {
            return abs((float)(mte()) / mte_max);
        }
        float uniform(float l, float u) {
            return rand() * (u - l) + l;
        }
        float normal(float mu, float sigma) {
            return std::normal_distribution<float>(mu, sigma)(mte) ;
        }
        template<class T>
        void shuffle(std::vector<T>& v) {
            std::shuffle(v.begin(), v.end(), mte);
        }

        template<class PopulationIterator, class SampleIterator, class Distance>
        void sample_snps(PopulationIterator start, PopulationIterator end, SampleIterator out, Distance n) {
            std::sample(start, end, out, n, mte);
        }
};

#endif