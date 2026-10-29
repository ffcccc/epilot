//common .h
#include <iostream>
#include <stdio.h>
#include <fstream>

// hdf .h
#include "hugedatafile/h5TPed_I.h"
#include "hugedatafile/hdfTPED.h"

// windows compat.
#ifdef WIN32
	#include <windows.h>
	//#define THDF5 h5TPED
	#define THDF5 hdfTPED
#else
	#define THDF5 hdfTPED
#endif

// cuda .h
#include "epi.h"


const int NumR = 10001;//20001;

// Host variables
float *h_Geno;
TF    *h_M, *h_M0, *h_M1, *h_CC;
T_Res *h_R;
unsigned h_NTest;

bool noprompt = false;
size_t Rsize = 0;

// var.
//extern T_GType *h_M, *h_M0, *h_M1, *h_CC;
//extern float *h_Geno;
//extern T_Res* h_R;

using namespace std;

// Functions
void Cleanup		(bool);
void RandomInit012	(T_GType*, int);
void RandomInit01	(T_GType*, int);
void ZeroInit		(float*, int);
void ZeroInit		(T_Res*, int);
void ParseArguments	(int, char**);

void matrixPartition(int nSamp, int nMarker, size_t maxAllocSz, vector<int> &parts);
void LocEpi (int i, int j, TF* M, TF* CC, float *d_Geno, int nSnp, int nSamp, float NAff, float NUnaff, bool diagonalMatrix, unsigned &ResIndex, int NumRes, T_Res *Res);
bool ctLocEpiAbs(const float CT[2][3][3], const float N[3], T_Res &RTmp);

// http://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
// ceiling(x,y) => (x + y - 1) / y  or faster...
inline int intCeiling(int x, int y){ return (1 + ((x - 1) / y)); };

//
void matrixPartition(const int nSamp, const int nMarker, const size_t maxAllocSz, vector<int> &parts){
	//
	size_t markerSz = nSamp * sizeof(T_GType);
	//
	size_t allocSz = min(nMarker * nSamp * sizeof(T_GType), maxAllocSz);
	//
	int nMarkerPart = intCeiling(allocSz, markerSz)-1;
	int n2MarkerPart = nMarkerPart / 2;
	size_t partSz = nMarkerPart * markerSz;
	int nBreaks = (nMarker / n2MarkerPart);
	parts.clear();
	parts.resize(nBreaks+2);
	parts[0] = 0;
	for(unsigned i=1; i<parts.size(); i++){
		parts[i] = min(parts[i-1] + n2MarkerPart, nMarker); 
	}
	int n = parts.size();
	for(int row=0; row<n-1; row++){
		for(int col=row; col<n-1; col++){
			cout << parts[row] << "-" << parts[row+1] << " vs " << parts[col] << "-" << parts[col+1] << endl;
		}
	}
}



// Host code
void printout(const int idxL, const int idxR, const vector<int> &partition, THDF5 *hf0, THDF5 *hf1){
	// Print result
	std::string ch1,ch2,s1,s2;
	THDF5 *hf;
	int ns0 = hf0->numSnps();
	int indsnp1(0), indsnp2(0);
	int nres = 0;
	ofstream outFile((hf0->filename() + ".cc.log").c_str(),ios_base::out|ios_base::app);
	outFile << "EdgeName\tCHR1\tSNP1\tCHR2\tSNP2\tX2 STAT\tfast-epi P\tOR\tSource\tTarget\n"; 
	//cout << "EdgeName\tCHR1\tSNP1\tCHR2\tSNP2\tX2 STAT\tfast-epi P\tOR\tSource\tTarget\n"; 
	
	for (int i = 0; i < NumR; ++i) {
		if(h_R[i].stat > 0.0001){
			hf = hf0;
			int n1 = partition[idxL] + h_R[i].s1;
			if(n1 < partition[idxL+1]){
				indsnp1 = n1;
			} else {
				indsnp1 = n1 - partition[idxL+1] + partition[idxR];
			}
			hf->getSNPChr (indsnp1, ch1);
			hf->getSNPName(indsnp1, s1);

			hf = hf0;
			int n2 = partition[idxL] + h_R[i].s2;
			if(n2 < partition[idxL+1]){
				indsnp2 = n2;
			} else {
				indsnp2 = n2 - partition[idxL+1] + partition[idxR];
			}
			hf->getSNPChr (indsnp2,	ch2);
			hf->getSNPName(indsnp2,	s2);
			
			outFile<< "R"<< nres++ <<"\t"<< ch1 <<"\t"<< s1 <<"\t"<< ch2 <<"\t"<< s2 <<"\t"<< h_R[i].stat <<"\t"<< h_R[i].p <<"\t"<< h_R[i].risk <<"\t"<< indsnp1 <<"\t"<< indsnp2 << endl;
			//cout << "R"<< nres++ <<"\t"<< ch1 <<"\t"<< s1 <<"\t"<< ch2 <<"\t"<< s2 <<"\t"<< h_R[i].stat <<"\t"<< h_R[i].p <<"\t"<< h_R[i].risk <<"\t"<< h_R[i].s1 <<"\t"<< h_R[i].s2 << endl;
		}
	}
}

int main1(THDF5 *hf) {
	bool freeall(false);
	int NSnp = hf->numSnps();	//1000;
	int NSamp = hf->numSamples();
	int NAff(0), NUnaff(0);
	// max memory allocable
    size_t MaxSize = 500000000;
	// alloc required memory
	size_t Msize = 0;
	// genotype test vector
	size_t Gsize = 0;
	// effective num. of snps per chunk
	int MSnp = 0;
	// list of chunk indexes	
	vector<int> partition;
	// compute chunk indexes
	matrixPartition(NSamp, NSnp, MaxSize, partition);
	// alloc size for 1 sample
	size_t Csize = 1 * NSamp * sizeof(T_GType);



	if(h_CC == 0)
		h_CC = (T_GType*)malloc(Csize);
	else
		h_CC = (T_GType*)realloc(h_CC, Csize);
    if (h_CC == 0) {
		Cleanup(true);
		assert(false);
	}

	// check-up
	CudaInfo();

	// Initialize pheno vectors
	int phe;
	for(int i=0; i<NSamp; i++){
		phe = -1;
		hf->getSamplePheno(i, phe);
		h_CC[i] = phe;
		if(phe==1)
			NAff++;
		else if(phe==0)
			NUnaff++;
	}

	int n = partition.size();
	for(int row=0; row<n-1; row++){
		// mark progression
		for(int k=0; k<row; k++){
			cout << "-";
		}
		for(int col=row; col<n-1; col++){
			//
			bool isDiagonal(col==row);
			// set the effective num of snps to test 
			MSnp = partition[row+1]-partition[row];		// snps from 1st chunk
			if(!isDiagonal)
				MSnp += partition[col+1]-partition[col];// snps from 2nd chunk

			// update alloc size to the effective num. of snps
			Msize = MSnp * Csize;
			// alloc size for prior genotypes results
			Gsize = MSnp * sizeof(float);

			// Allocate input vectors h_A and h_B in host memory
			if(h_M == 0)
				h_M = (T_GType*)malloc(Msize);
			else
				h_M = (T_GType*)realloc(h_M, Msize);
			if (h_M == 0) {
				cout << "Alloc issue";
				Cleanup(true);
				assert(false);
			}
			
			// Initialize g-type test vector
			if(h_Geno == 0)
				h_Geno = (float*)malloc(Gsize);
			else
				h_Geno = (float*)realloc(h_Geno, Gsize);
			if (h_Geno == 0) {
				cout << "Alloc issue";
				Cleanup(true);
				assert(false);
			}

			int pos_M = 0;	// 0 based index in the new matrix h_M
			for(int i=partition[row]; i<partition[row+1]; i++){
				string chr("");	
				hf->getSNPChr(i, chr);
				if((chr != "6") && (chr != "23")){
					hf->getSnpMem(i, &h_M[pos_M*NSamp]);
					hf->getSNPScore(i, h_Geno[pos_M]);
				}
				pos_M++;
			}

			if(! isDiagonal){
				for(int i=partition[col]; i<partition[col+1]; i++){
					string chr("");	
					hf->getSNPChr(i, chr);
					if((chr != "6") && (chr != "23")){
						hf->getSnpMem(i, &h_M[pos_M*NSamp]);
						hf->getSNPScore(i, h_Geno[pos_M]);
					}
					pos_M++;
				}
			}

			// mark progression
			cout << "*";
			/*
			unsigned h_RInd;
			if(isDiagonal){
				// nsnps * nsnps
				// griglia N/2 x N --> matrice triangolare: sottomatrici sulla diagonale
				//dim3 blocksPerGrid( (nSnp/2)/threadsPerBlock.x, nSnp/threadsPerBlock.y, 1);
				for(int x=0; x<NSnp/2; x++){
					for(int y=0; y<NSnp; y++){
						LocEpi(x, y, h_M, h_CC, h_Geno, MSnp, NSamp, NAff, NUnaff, isDiagonal, h_RInd, (Rsize/sizeof(T_Res))-1, h_R);
					}
				}
			} else {
				// griglia N x N --> matrice completa
				//dim3 blocksPerGrid( (nSnp/2)/threadsPerBlock.x, (nSnp/2)/threadsPerBlock.y, 1);
				for(int x=0; x<NSnp/2; x++){
					for(int y=0; y<NSnp/2; y++){
						LocEpi(x, y, h_M, h_CC, h_Geno, MSnp, NSamp, NAff, NUnaff, isDiagonal, h_RInd, (Rsize/sizeof(T_Res))-1, h_R);
					}
				}
			}
			*/
			// run computation
			kernelLaunch(h_M, Msize, h_CC, Csize, h_Geno, Gsize, h_R, Rsize, MSnp, NSamp, float(NAff), float(NUnaff), isDiagonal, &h_NTest);
			//kernelLaunch(h_M, AllocSz, h_CC, Csize, h_R, Rsize, NSnp, NSamp, float(NAff), float(NUnaff));
			// print a report log


			printout(row, col, partition, hf, 0);
			Cleanup(freeall);
		}
		cout << endl;
	}

	return 1;
}


void Cleanup(bool freeall){

	CudaCleanup();

	// snp data
	if (h_M){
        free(h_M);
		h_M = 0;
	}

	// geno test
	if (h_Geno){
        free(h_Geno);
		h_Geno = 0;
	}

	/*if (freeall && h_M0){
        free(h_M);
		h_M0 = 0;
	}
	if (freeall && h_M1){
        free(h_M);
		h_M1 = 0;
	}
	*/
	// pheno
	if (freeall && h_CC){
        free(h_CC);
		h_CC = 0;
	}

	_heapmin();
}

// Allocates an array with random T_GType entries.
void RandomInit012(T_GType* data, int n)
{
	float tmp;
	for (int i = 0; i < n; ++i){
        //data[i] = rand() / (T_GType)RAND_MAX;
		tmp = rand() / (float)RAND_MAX;
		if(tmp < 0.3)
			data[i] = 0;
		else if(tmp < 0.6)
			data[i] = 1;
		else
			data[i] = 2;
	}
}

// Allocates an array with random T_GType entries.
void RandomInit01(T_GType* data, int n)
{
	float tmp;
	for (int i = 0; i < n; ++i){
		tmp = rand() / (float)RAND_MAX;
		if(tmp < 0.5)
			data[i] = 0;
		else
			data[i] = 1;
	}
}

void ZeroInit(float* data, int n)
{
	for (int i = 0; i < n; ++i){
        data[i] = 0;
	}
}

void ZeroInit(T_Res* data, int n)
{
	for (int i = 0; i < n; ++i){
		data[i].stat = 0.;
		data[i].risk = 1.;
		data[i].s1 = -1;
		data[i].s2 = -1;
	}
}

// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i)
        if (strcmp(argv[i], "--noprompt") == 0 ||
			strcmp(argv[i], "-noprompt") == 0) 
		{
            noprompt = true;
            break;
        }
}

// 
// cad\cad-58C_21_qc-rs.h5
// cad\cad-58C_21_qc-rs.h5 cad\cad-58C_22_qc-rs.h5 
int main(int argc, char** argv){
	THDF5 *hf0;//, *hf1;
	h_M = 0;
	h_M0 = 0;
	h_M1 = 0;
	h_R = 0;

	ParseArguments(argc, argv); 
	cout << "epistasis test...\n";
	for(int i=0; i<argc; i++)
		cout << argv[i] << "\n";

	Rsize = NumR * sizeof(T_Res);
	h_R = (T_Res*)malloc(Rsize);
	if (h_R == 0) {
		assert(false);
	}
	ZeroInit(h_R, NumR);

	vector<string> file;
	if(argc == 1){
		// forced list of filename
		for(unsigned i=0; i<file.size(); i++){
			hf0 = new THDF5(file[i]);
			main1(hf0);
			//main0(hf0);
			//for(int j=i+1; j<file.size(); j++){
			//	hf1 = new THDF5(file[j]);
			//	main2(hf0, hf1);
			//	delete hf1;
			//}
			delete hf0;
		}
	}

	if(argc == 2){
		hf0 = new THDF5(argv[1]);
		main1(hf0);
		delete hf0;
	}

	Cleanup(true);
	if (h_R){
        free(h_R);
		h_R = 0;
	}
	
	if (!noprompt) {
        printf("\nPress ENTER to exit...\n");
        fflush( stdout);
        fflush( stderr);
        getchar();
    }
    exit(0);
}

bool ctLocEpiAbs(const float CT[2][3][3], const float N[3], T_Res &RTmp){
	bool found = false;
	float A,B,C,D,Chi2;
	for(int r=0;r<3;r++){
		for(int c=0;c<3;c++){
			A = CT[Aff][r][c];
			B = CT[Unaff][r][c];
			C = N[Aff] - A;
			D = N[Unaff] - B;
			if((A>0)&&(B>0)&&(C>0)&&(D>0)){
				// Chi Quadro
				float temp = (A*D)-(C*B);
				Chi2 = (temp * temp * N[All]) / ((A+C)*(A+B)*(C+D)*(B+D));
				// Chi quadro corretto Yates
				//float N =A+B+C+D //float N = nCC - NA[Aff] - NA[Unaff] 
				//float Chi2Yates=(pow(fabs((A*D)-(C*B))-(N/2),2)*N)/((A+C)*(A+B)*(C+D)*(B+D));
				if(Chi2 > CHI10E9){
					found = true;
					RTmp.stat = Chi2;
					RTmp.risk = (A*D)/(B*C); 
				}
			}
		}
	}
	return found;
}

void LocEpi (int i, int j, TF* M, TF* CC, float *d_Geno, int nSnp, int nSamp, float NAff, float NUnaff, bool diagonalMatrix, unsigned &ResIndex, int NumRes, T_Res *Res)
{
	float CT[2][3][3];
	float NA[2];
	float p(1);

	TF* S1 = 0;
	TF* S2 = 0;
    //int i = blockDim.x * blockIdx.x + threadIdx.x;
	//int j = blockDim.y * blockIdx.y + threadIdx.y;
	if(diagonalMatrix){
		if(i > j){
			i = nSnp-1-i;
			j = nSnp-1-j;
		}
	} else {
		j += nSnp/2;
	}

	S1 = M+(i*nSamp);
	S2 = M+(j*nSamp);
	
	if(S1 && S2){
		// init cont. tables
		for(int s=0;s<2;s++){
			NA[s] = 0;
			for(int r=0;r<3;r++){
				for(int c=0;c<3;c++){
					CT[s][r][c]=0;
				}
			}
		}
		// build cont. tables
		for(int k=0;k<nSamp;k++){
			if((S1[k]<0) || (S2[k]<0))	NA[CC[k]]++;
			else						CT[CC[k]][int(S1[k])][int(S2[k])]++;
		}

		// screening epistasis
		p = 0.;
		//p = fastepi(CT);
	
		// deep epistasis
		if(p < P_Epi){
			float N[3];	// corregge il parametro di casi e controlli con il conteggio degli effettivi
			N[Aff]		= NAff-NA[Aff];
			N[Unaff]	= NUnaff-NA[Unaff];
			N[All]		= N[Aff] + N[Unaff];
			bool found(false);
			T_Res RTmp;
			RTmp.p = p;
			RTmp.stat = 0;
			found = ctLocEpiAbs(CT,N,RTmp);
			if(found && (RTmp.stat > d_Geno[i]) && (RTmp.stat > d_Geno[j])){
				RTmp.s1 = i;
				RTmp.s2 = j;
				// pos 0 = Err. | pos [1-NR] = valid | initially RIndex[0] = 1;
				unsigned pos = ++ResIndex;
				if(pos > 0){
					Res[pos] = RTmp;
				}
			}
		}
	}
}
