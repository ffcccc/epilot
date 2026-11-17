// Core types shared by the CPU and GPU back ends.
// Anything marked EPI_HD is compiled for host and device (under nvcc), so the test
// statistics and the per-pair logic exist once and run identically on both paths.
#ifndef EPI_CORE_H

#define EPI_CORE_H
#include <cstddef>
#include <cstdint>
// #include <algorithm>
// #include <stdexcept>
#include <string>
#include <vector>

#ifdef __CUDACC__
    #define EPI_HD __host__ __device__
#else
    #define EPI_HD
#endif

namespace epi {

// #define UNAFF	0
// #define AFF		1
// #define All		2
//(1-pchisq(50,1))*6.7e10 = alpha 0.1
#define CHI10E9 25
// 0.001 N(0,1)
#define EpiScreeningTH 3
// or
#define P_Epi 0.01
//
#define MAX_TEST 100000000000

//Chi2   P
//7.8 -> 0.005
//10  -> 0.001
//25  -> 10E-7

// h5 file format
// typedef char T_GType;
typedef uint8_t T_GType;

typedef struct {
	float stat;
	float risk;
	int g_row;
	int g_col;
} T_Cel;

// One retained SNP pair: Pearson statistic, cell odds ratio, screening value,
// and tile-local SNP indices. p currently holds z, not a calibrated p-value.
typedef struct {
	float stat;
	float risk;
	float p;
	int s1;
	int s2;
	//float A;
	//float B;
	//float C;
	//float D;
	//T_Cel geno[9]; 
} T_Res;

// SNP-major genotype storage: 0, 1, 2; negative values denote missing data.
// Plain char must be signed for the kernel's missing-value check to work.
#define TF char 

// Kernel code defined in epi.cu ver. 0.1.0:
// void CudaInfo(void);
// void CudaCleanup(void);
// void kernelLaunch(TF *h_M, int size, TF *h_CC, int Csize, float *h_Geno, int Gsize, T_Res *h_R, int Rsize, int nSnp, int nSamp, float NAff, float NUnaff, bool diagonalMatrix, unsigned *h_NTest);
//void kernelLauch2(TF *h_M1, int size1, TF *h_M2, int size2, TF *h_CC, int Csize, T_Res *h_R, int Rsize, int nSnp1, int nSnp2, int nSamp, float NAff, float NUnaff);
//void kernelLoad(T_GType *h_TMP, int size, int index);
//void kernelInit(int size, TF *h_CC, int Csize, T_Res *h_R, int Rsize);
//void kernelRun(int Rsize, int nSnp, int nSamp, float NAff, float NUnaff);


// Arithmetic type of the test statistics. double by default so that CPU and GPU
// give the same answers; compile with -DEPI_REAL_FLOAT for faster GPU arithmetic.
#ifdef EPI_REAL_FLOAT
    typedef float real;
#else
    typedef double real;
#endif

// Group indices into CT[group] and N[group]. Genotype codes are minor-allele
// counts 0..2; MISSING is any negative value.
enum : int { UNAFF = 0, AFF = 1, ALL = 2 };

// Parameters only some tests use (e.g. the screen of the two-stage test).
struct TestOptions {
    real screen_min;  // FastEpistasis |z| needed before Chi2 is run (two-stage test)
};

// Output of one epistatic test on one SNP pair.
struct TestOut {
    real stat;     // test statistic; 0 when the test reports nothing
    real risk;     // odds ratio of the reported cell
    int32_t row;   // reported joint-genotype cell (SNP1 genotype), -1 if none
    int32_t col;   // reported joint-genotype cell (SNP2 genotype), -1 if none
};

// One significant pair, with global SNP indices s1 < s2.
struct Hit {
    uint32_t s1, s2;
    real stat, risk;
    int32_t row, col;
};

// Everything a pair evaluation needs, as plain data so it can be passed to a kernel.
// A tile holds two chunks of GS SNPs: rows [0, GS) are chunk A (SNPs snpA + r) and
// rows [GS, 2GS) are chunk B (SNPs snpB + c). For a diagonal tile, chunk B is chunk A
// and only pairs with c > r are tested.
struct TileView {
    const int8_t* geno;    // 2*GS rows x nSamp, row-major (SNP-major)
    const uint8_t* valid;  // 2*GS: 0 for padding and excluded SNPs
    const real* score;     // 2*GS: univariate score; a pair must beat both
    const uint8_t* cc;     // nSamp: UNAFF or AFF
    int nSamp;
    int GS;
    bool diagonal;
    real nAff, nUnaff;     // samples with known phenotype, per group
    real min_stat;         // a pair is reported only if stat > min_stat
    TestOptions opt;
    uint32_t snpA, snpB;   // global index of the first SNP of each chunk
};

// Read-only access to a genotype dataset. Implement this over your file reader.
class SnpSource {
public:
    virtual ~SnpSource() {}
    virtual size_t numSnps() const = 0;
    virtual size_t numSamples() const = 0;
    virtual int phenotype(size_t sample) const = 0;             // 1 affected, 0 unaffected, other = unknown
    virtual std::string chromosome(size_t snp) const = 0;
    virtual int8_t genotype(size_t i, size_t j) const = 0;
    virtual void genotypes(size_t snp, int8_t* out) const = 0;  // numSamples values in {-1,0,1,2}
    virtual float score(size_t snp) const = 0;                  // univariate score
};

struct ScanConfig {
    std::string test = "Chi2";                 // see with_test() for names
    real min_stat = 0;                         // report pairs with stat > min_stat
    TestOptions opt = {3.0};                   // screen threshold of "fastepistasis+Chi2"
    size_t GS = 1024;                          // SNPs per chunk
    std::vector<std::string> excluded_chromosomes = {"6", "23"};
    size_t max_tile_bytes = 500000000;         // host memory budget for one tile
};

}  // namespace epi
#endif  // EPI_CORE_H