#ifndef __CONFIG_H
#define __CONFIG_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include "common.h"



template <typename T>
T clip(const T& n, const T& lower, const T& upper) {
  return std::max(lower, std::min(n, upper));
}

template<class T>
inline constexpr T pow(const T base, unsigned const exponent)
{
    return (exponent == 0)     ? 1 :
           (exponent % 2 == 0) ? pow(base, exponent/2)*pow(base, exponent/2) :
           base * pow(base, (exponent-1)/2) * pow(base, (exponent-1)/2);
}

class Config {
    std::string geno_input_path = "";
    std::string cov_input_path = "";
    std::string output_path = "";
    size_t N = 0, K = 0, G = 0;
    int argc = 0;

    public:
        bool _continue = false;
        bool _pretrain = true;
        bool se_only = false;
        bool bootstrap = false;
        int bootstrap_size = -1;
        size_t newton_iter = 10, em_iter = 30, epochs = 1, batch_size = 1000;
        void parse_config(int argc, const char** argv);
        void parse_param(int* current_loc, const char** argv);
        std::string get_geno_input_path() const { return geno_input_path; };
        std::string get_cov_input_path() const { return cov_input_path; };
        std::string get_output_path() const { return output_path; };
        inline size_t get_N() const { return N; };
        inline size_t get_K() const { return K; };
        inline size_t get_G() const { return G; };
        inline size_t get_batch_size() const { return batch_size; };
        inline size_t get_newton_iter() const { return newton_iter; };
        inline size_t get_em_iter() const { return em_iter; };
        inline size_t get_epochs() const { return epochs; };
        static void show_available_options();
};

void
Config::show_available_options() {
    const char* help = 
    "Usage: mesa -i INPUT [--cov COV_FILE] -N NUM_INDIVIDUALS -K NUM_ANCESTRIES -G NUM_SNPS -o OUTPUT_PREFIX [OPTION]...\n"
    "\n"
    "Required arguments:\n"
    "   -i, --in INPUT              Path of genotype matrix file (String). \n"
    "                               Accepted file type: tsv, ttsv, bed\n"
    "                               e.g. -i /home/User/genotype.bed\n"
    "   -N NUM_INDIVIDUALS          Number of individuals to be tested\n"
    "                               (Positive integer)\n"
    "   -K NUM_ANCESTRIES           Number of ancestries to be fitted\n"
    "                               (Positive integer)\n"
    "   -G NUM_SNPS                 Number of SNPs to be tested (Positive\n"
    "                               integer)\n"
    "   -o, --out OUTPUT_PREFIX     Prefix of path of output file\n"
    "                               (String). For example, option\n"
    "                               '-o /home/User/output' will produce\n"
    "                               files such as '/home/User/output_p.tsv'\n"
    "\n"
    "Optional arguments:\n"
    "   --cov COV_FILE              Path of phenotype matrix file (String)\n"
    "   --batch-size                Number of subsampled SNPs (Positive\n"
    "                               integer, 1000\n"
    "   --newton-iter X             Maximum number of Newton's steps = X\n"
    "                               (Positive integer, default: 10)\n"
    "   --em-iter X                 Maximum number of E-steps and M-steps = X\n"
    "                               (Positive integer, default: 30)\n"
    "   --epochs X                  Number of passes through whole data = X\n"
    "                               set (Positive integer, default: 1)\n"
    "   --continue                  Whether to use existing estimates\n"
    "                               extracted from OUTPUT_PREFIX_p.tsv, etc\n"
    "   --no-pretrain               Start training with no warm-up\n"
    "   --se-only                   Calculate standard errors using\n"
    "                               existing estimates extracted from\n"
    "                               OUTPUT_PREFIX_p.tsv,\n"
    "                               OUTPUT_PREFIX_q.tsv and \n"
    "                               OUTPUT_PREFIX_effect.tsv\n"
    "\n"
    ;
    std::cout << help << std::endl;
}

void
Config::parse_param(int* current_loc, const char** argv) {
    const char* curr_argv = argv[*current_loc];

    if (std::strcmp(curr_argv, "-i") == 0 || std::strcmp(curr_argv, "--in") == 0) {
        if (*current_loc < argc) {
            geno_input_path = argv[(*current_loc) + 1];
            std::cout << "geno_input_path: " << geno_input_path << std::endl;
        }
    } else if (std::strcmp(curr_argv, "--cov") == 0) {
        if (*current_loc < argc) {
            cov_input_path = argv[(*current_loc) + 1];
            std::cout << "cov_input_path: " << cov_input_path << std::endl;
        }
    } else if (std::strcmp(curr_argv, "-K") == 0) {
        if (*current_loc < argc) {
            K = std::stoul(argv[(*current_loc) + 1]);
            std::cout << "K: " << K << std::endl;
        }
    } else if (std::strcmp(curr_argv, "-G") == 0) {
        if (*current_loc < argc) {
            G = std::stoul(argv[(*current_loc) + 1]);
            std::cout << "G: " << G << std::endl;
        }
    } else if (std::strcmp(curr_argv, "-N") == 0) {
        if (*current_loc < argc) {
            N = std::stoul(argv[(*current_loc) + 1]);
            std::cout << "N: " << N << std::endl;     
        }
    } else if (std::strcmp(curr_argv, "--batch-size") == 0) {
        if (*current_loc < argc) {
            batch_size = std::stoul(argv[(*current_loc) + 1]);
            std::cout << "Number of subsampled SNPs: " << batch_size << std::endl;     
        }
    } else if (std::strcmp(curr_argv, "--newton-iter") == 0) {
        if (*current_loc < argc) {
            newton_iter = std::stoul(argv[(*current_loc) + 1]);
            std::cout << "Maximum number of Newton's steps: " << newton_iter << std::endl;     
        }
    } else if (std::strcmp(curr_argv, "--em-iter") == 0) {
        if (*current_loc < argc) {
            em_iter = std::stoul(argv[(*current_loc) + 1]);
            std::cout << "Maximum number of E-steps and M-steps: " << em_iter << std::endl;     
        }
    } else if (std::strcmp(curr_argv, "--epochs") == 0) {
        if (*current_loc < argc) {
            epochs = std::stoul(argv[(*current_loc) + 1]);
            std::cout << "Number of epochs: " << epochs << std::endl;     
        }
    } else if (std::strcmp(curr_argv, "-o") == 0 || std::strcmp(curr_argv, "--out") == 0) {
        if (*current_loc < argc) {
            output_path = argv[(*current_loc) + 1];
            std::cout << "output_path: " << output_path << std::endl;     
        }
    } else if (std::strcmp(curr_argv, "--continue") == 0) {
        _continue = true;
        std::cout << "continue: true" << std::endl;
        *current_loc -= 1;
    } else if (std::strcmp(curr_argv, "--no-pretrain") == 0) {
        _pretrain = false;
        std::cout << "pretrain: no" << std::endl;
        *current_loc -= 1;
    } else if (std::strcmp(curr_argv, "--se-only") == 0) {
        se_only = true;
        _continue = true;
        std::cout << "se_only: true" << std::endl;
        *current_loc -= 1;
    } else {
        std::cerr << "Invalid argument: " << argv[*current_loc] << std::endl;
        throw std::exception();
    }
    *current_loc += 2;
}

void
Config::parse_config(int argc, const char** argv) {
    this->argc = argc;
    int i = 1;
    while (i < argc) {
        std::string argv_str(argv[i]);
        if (strcmp(argv_str.substr(0, 1).c_str(), "-") == 0)
            parse_param(&i, argv);
        else
            i++;
    }
    if (N < 1) throw std::exception();
    if (K < 1 || K > MAX_K) throw std::exception();
    if (G < 1) throw std::exception();
}



class GenoDataReader {
    std::string input_path = "";
    GenoData data;
    bool available = false;

    public:
        size_t shape[2] = {};

        GenoDataReader& open(const Config& c) {
            available = c.get_geno_input_path().length() > 0;
            if (available) {
                this->input_path = c.get_geno_input_path();
                std::ifstream ifs(c.get_geno_input_path());
                
                std::string line;
                uint8_t buffer;
                size_t row = 0, col = 0;

                size_t N = c.get_N();
                size_t G = c.get_G();

                data = new uint8_t[N * G];

                if (data != nullptr) {
                    while (getline(ifs, line) && row < N) {
                        std::istringstream iss(line, std::ios_base::in);
                        
                        col = 0;
                        while ((iss >> std::dec >> buffer) && col < G) {
                            buffer -= 48;
                            if (buffer > 2 || buffer != 9) throw std::exception();
                            data[row * G + col] = buffer;
                            ++col;
                        }
                        if (col != G) {
                            std::cerr << "Invalid number of columns" << std::endl;
                            throw std::exception();
                        }
                        ++row;
                    }
                    if (row != N) {
                        std::cerr << "Invalid number of rows" << std::endl;
                        throw std::exception();
                    }
                    ifs.close();
                } else {
                    std::cerr << "Cannot allocate memory for genotype data" << std::endl;
                    throw std::exception();
                }

                shape[0] = N;
                shape[1] = G;
            }
            return *this;
        }
        
        GenoDataReader& open_t(const Config& c) {
            available = c.get_geno_input_path().length() > 0;
            if (available) {
                this->input_path = c.get_geno_input_path();
                std::ifstream ifs(c.get_geno_input_path());
                
                std::string line;
                uint8_t buffer;
                size_t row = 0, col = 0;
                size_t N = c.get_N(), G = c.get_G();

                data = new uint8_t[N * G];

                if (data != nullptr) {
                    while (getline(ifs, line) && row < G) {
                        std::istringstream iss(line, std::ios_base::in);
                        
                        col = 0;
                        while ((iss >> std::dec >> buffer) && col < N) {
                            buffer -= 48;
                            if (buffer > 2 || buffer != 9) throw std::exception();
                            data[col * G + row] = buffer;
                            ++col;
                        }
                        if (col != N) {
                            std::cerr << "Invalid number of columns" << std::endl;
                            throw std::exception();
                        }
                        ++row;
                    }
                    if (row != G) {
                        std::cerr << "Invalid number of rows" << std::endl;
                        throw std::exception();
                    }
                    ifs.close();
                } else {
                    std::cerr << "Cannot allocate memory for genotype data" << std::endl;
                    throw std::exception();
                }

                shape[0] = N;
                shape[1] = G;
            }
            return *this;
        }

        GenoDataReader& open_bed(const Config& c) {
            available = c.get_geno_input_path().length() > 0;
            if (available) {
                this->input_path = c.get_geno_input_path();
                std::ifstream ifs(c.get_geno_input_path(), std::ios::in | std::ios::binary);
                
                std::string line;
                size_t i = 0, l = 0;
                size_t N = c.get_N(), G = c.get_G();
                char magic_number[4] = "";

                data = new uint8_t[N * G];
                uint8_t bed_buffer[BED_READ_SIZE_BYTES];

                if ( data != nullptr && ifs.good() ) {
                    ifs.read(magic_number, 3);
                    bool valid = !strcmp(magic_number, "\x6c\x1b\x01");
                    if (!valid)
                        throw InvalidGenoFileError();
                    std::cout << "Reading bed file..." << std::endl;

                    do {
                        ifs.read((char*)bed_buffer, BED_READ_SIZE_BYTES);
                        for (std::streamsize c = 0; c < ifs.gcount(); ++c) {
                            // std::cout << "c: " << c << std::endl;
                            for (uint8_t pos = 0; pos < 4; ++pos) {
                                uint8_t bits = bed_buffer[c] & 0b11;
                                switch(bits) {
                                    case 0b00:
                                        data[i * G + l] = 0b00;
                                        break;
                                    case 0b01:
                                        data[i * G + l] = 0b1001;
                                        break;
                                    case 0b10:
                                        data[i * G + l] = 0b01;
                                        break;
                                    case 0b11:
                                        data[i * G + l] = 0b10;
                                        break;
                                    default:
                                        std::cerr << "Does not support unknown genotype: " << (int)bits << " at i=" << i << ",l=" << l << std::endl;
                                        throw InvalidGenoFileError();
                                }
                                bed_buffer[c] >>= 2;
                                if (++i >= N) {
                                    i = 0;
                                    ++l;
                                    if (c < ifs.gcount()) break;
                                }
                                if (l >= G) {
                                    if ( ifs.eof() || c == ifs.gcount() - 1 ) break;
                                    else {
                                        std::cerr << "Bad bed file" << std::endl;
                                        throw InvalidGenoFileError();
                                    }
                                }
                            }
                        }
                    } while ( !ifs.eof() );
                    ifs.close();
                    // for (size_t i = N - 6; i < N; ++i) {
                    //     for (size_t l = G - 6; l < G; ++l) {
                    //         std::cout << (int)data[i * G + l];
                    //         std::cout << (l + 1 < G ? " " : "\n");
                    //     }
                    // }
                    // std::cout << std::endl;
                } else {
                    std::cerr << "Cannot allocate memory for genotype data" << std::endl;
                    throw std::exception();
                }

                shape[0] = N;
                shape[1] = G;
            }
            return *this;
        }

        inline GenoData get_data() { return available ? data : nullptr; }

        void close() {
            delete[] data;
            data = nullptr;
        }

        inline std::string& get_input_path() { return input_path; };
};

class CovDataReader {
    std::string input_path = "";
    CovData data;
    bool available = false;
    
    public:
        size_t shape[2] = {};
        // size_t tmp_n = 0;

        CovDataReader& open(const Config& c) {
            available = c.get_cov_input_path().length() > 0;
            if (available) {
                this->input_path = c.get_cov_input_path();
                std::ifstream ifs(c.get_cov_input_path());
                
                std::string line;
                MESA_NUMERIC_TYPE buffer;
                size_t N = c.get_N();
                size_t row = 0;
                size_t prev_cov_num = 0;

                data = new CovRow[N];

                while (getline(ifs, line) && row < N) {
                    std::istringstream iss(line, std::ios_base::in | std::ios_base::trunc);
                    
                    while (iss >> buffer) {
                        data[row].push_back(buffer);
                    }
                    if (row > 1 && data[row].size() != prev_cov_num) {
                        std::cerr << "Invalid number of covariates" << std::endl;
                        throw std::exception();
                    }
                    prev_cov_num = data[row].size();
                    ++row;
                }
                if (row != N) {
                    std::cerr << "Invalid number of rows" << std::endl;
                    throw std::exception();
                }
                ifs.close();

                shape[0] = N;
                shape[1] = data[0].size();
            }
            return *this;
        }

        inline CovData get_data() { return available ? data : nullptr; }

        void close() {
            // for (auto it = data->begin(); it != data->end(); it++)
            //     delete *it;
            // std::vector<CovRow*>().swap(*data);
            delete[] data;
            data = nullptr;
        }

        inline std::string& get_input_path() { return input_path; };
};

class ResultDataReader {
    std::string input_path = "";
    size_t N = 0, K = 0, G = 0, cov_num = 0;
    bool available = false;
    MESA_NUMERIC_TYPE* theta;
    size_t theta_size;

    public:
        ResultDataReader(const Config& c, size_t _cov_num) : N(c.get_N()), K(c.get_K()), G(c.get_G()), cov_num(_cov_num) {
            theta_size = K * G * (cov_num + 1) + N * K;
            theta = new MESA_NUMERIC_TYPE[theta_size];
            available = c.get_output_path().length() > 0;

            if (available) {
                input_path = c.get_output_path();
                std::string p_path = input_path + "_p.tsv";
                std::string q_path = input_path + "_q.tsv";
                std::string cov_path = input_path + "_effect.tsv";

                std::string line;
                MESA_NUMERIC_TYPE buffer;

                size_t row, col;

                // Read estimated population allele frequency
                std::ifstream ifs(p_path);
                if (ifs.is_open()) {
                    row = 0;
                    std::cout << "Reading " << p_path << "..." << std::endl;

                    while (ifs.good() && getline(ifs, line)) {
                        std::istringstream iss(line, std::ios_base::in | std::ios_base::trunc);
                        col = 0;
                        while (iss >> buffer) {
                            if (row < 10)
                                std::cout << buffer << " ";
                            theta[(col * G + row) * (cov_num + 1)] = buffer;
                            ++col;
                        }
                        if (row < 10)
                            std::cout << std::endl;
                        ++row;
                    }
                } else {
                    available = false;
                }
                ifs.close();
                
                // Read estimated ancestral proportions
                ifs.open(q_path);
                if (ifs.is_open()) {
                    row = 0;
                    std::cout << "Reading " << q_path << "..." << std::endl;

                    while (ifs.good() && getline(ifs, line)) {
                        std::istringstream iss(line, std::ios_base::in | std::ios_base::trunc);
                        col = 0;
                        while (iss >> buffer) {
                            if (row < 10)
                                std::cout << buffer << " ";
                            theta[K * G * (cov_num + 1) + row * K + col] = buffer;
                            ++col;
                        }
                        if (row < 10)
                            std::cout << std::endl;
                        ++row;
                    }
                } else {
                    available = false;
                }
                ifs.close();

                // Read estimated population effects
                ifs.open(cov_path);
                if (cov_num > 0 && ifs.is_open()) {
                    row = 0;
                    std::cout << "Reading " << cov_path << "..." << std::endl;

                    while (ifs.good() && getline(ifs, line)) {
                        std::istringstream iss(line, std::ios_base::in | std::ios_base::trunc);
                        col = 0;
                        while (iss >> buffer) {
                            if (row < 10)
                                std::cout << buffer << " ";
                            theta[(col/cov_num * G + row) * (cov_num + 1) + 1 + (col % cov_num)] = buffer;
                            ++col;
                        }
                        if (row < 10)
                            std::cout << std::endl;
                        ++row;
                    }
                }
                ifs.close();
            }
        }

        inline MESA_NUMERIC_TYPE* get_data() const { return available ? theta : nullptr; }

        ~ResultDataReader() {
            delete[] theta;
            theta = nullptr;
        }

        inline std::string& get_input_path() { return input_path; };
        inline size_t get_theta_size() { return theta_size; };
        inline int get_N() const { return N; };
        inline int get_K() const { return K; };
        inline int get_G() const { return G; };
        inline int get_cov_num() const { return cov_num; };
};

#endif