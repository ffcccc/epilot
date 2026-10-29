#define Unaff	0
#define Aff		1
#define All		2
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

// Kernel code
void CudaInfo(void);
void CudaCleanup(void);
void kernelLaunch(TF *h_M, int size, TF *h_CC, int Csize, float *h_Geno, int Gsize, T_Res *h_R, int Rsize, int nSnp, int nSamp, float NAff, float NUnaff, bool diagonalMatrix, unsigned *h_NTest);
//void kernelLauch2(TF *h_M1, int size1, TF *h_M2, int size2, TF *h_CC, int Csize, T_Res *h_R, int Rsize, int nSnp1, int nSnp2, int nSamp, float NAff, float NUnaff);
//void kernelLoad(T_GType *h_TMP, int size, int index);
//void kernelInit(int size, TF *h_CC, int Csize, T_Res *h_R, int Rsize);
//void kernelRun(int Rsize, int nSnp, int nSamp, float NAff, float NUnaff);