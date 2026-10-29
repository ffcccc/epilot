// Includes
//#include "cuPrintf.cuh"
// cuda
#include <cutil_inline.h>
#include "epi.h"

unsigned initval(1);

// Device Variables
__device__ TF* d_M;
__device__ float* d_Geno;
__device__ TF* d_CC;
__device__ T_Res* d_R;
__device__ unsigned *d_RInd, *d_NTest;

// Device code
__global__ void VecAdd(const TF* A, const TF* B, TF* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
	//cuPrintf(�Thread is: %d\n�, i);
}

__constant__ float sqrt2pi = 2.50662827463;

// CT[phenotype][SNP1 genotype][SNP2 genotype] contains complete-pair counts.
// Compare allele-product log odds ratios between cases and controls.
// Known defect: the inner declaration of z shadows the returned value (always 0).
__device__ float fastepi(const float CT[2][3][3]){
	//float p(1);
	float z(0);
	float A = 4*CT[Aff][0][0]	+ 2*CT[Aff][0][1]	+ 2*CT[Aff][1][0]	+ CT[Aff][1][1];
	float a = 4*CT[Unaff][0][0]	+ 2*CT[Unaff][0][1]	+ 2*CT[Unaff][1][0]	+ CT[Unaff][1][1];
	float B = 4*CT[Aff][0][2]	+ 2*CT[Aff][0][1]	+ 2*CT[Aff][1][2]	+ CT[Aff][1][1];
	float b = 4*CT[Unaff][0][2]	+ 2*CT[Unaff][0][1]	+ 2*CT[Unaff][1][2]	+ CT[Unaff][1][1];
	float C = 4*CT[Aff][2][0]	+ 2*CT[Aff][2][1]	+ 2*CT[Aff][1][0]	+ CT[Aff][1][1];
	float c = 4*CT[Unaff][2][0]	+ 2*CT[Unaff][2][1]	+ 2*CT[Unaff][1][0]	+ CT[Unaff][1][1];
	float D = 4*CT[Aff][2][2]	+ 2*CT[Aff][2][1]	+ 2*CT[Aff][1][2]	+ CT[Aff][1][1];
	float d = 4*CT[Unaff][2][2]	+ 2*CT[Unaff][2][1]	+ 2*CT[Unaff][1][2]	+ CT[Unaff][1][1];
	if((A>0)&&(a>0)&&(B>0)&&(b>0)&&(C>0)&&(c>0)&&(D>0)&&(d>0)){
		float varCa = 1.0/A + 1.0/D + 1.0/B + 1.0/C;
		float varCo = 1.0/a + 1.0/d + 1.0/b + 1.0/c;
		float orCa	= (A*D)/(B*C);
		float orCo	= (a*d)/(b*c);
		float z = fabs( (log(orCa) - log(orCo)) / sqrt(varCa + varCo) );
	}
	return z;
	//	// P = normdist(-z)*2
	//	
	//	float t0, z1, p0 ;
	//	t0 = 1 / (1 + 0.2316419 * z);
	//	z1 = exp(-0.5 * z*z ) / sqrt2pi;
	//	p0 = z1 * t0 * (0.31938153 + t0 * (-0.356563782 + t0 * (1.781477937 + t0 * (-1.821255978 + 1.330274429 * t0))));
	//	if(z < 0)
	//		p = 1-p0;
	//	else
	//		p = p0;
	//	p = p*2;
	//}
	//return p;
}

// Test each joint-genotype cell against all other cells (Pearson 2x2 test).
// Only positive cells are tested; the last passing cell, not the maximum, is kept.
__device__ bool ctEpiAbs(const float CT[2][3][3], const float N[3], T_Res &RTmp){
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

// Unused alternative: compare each cell with the double-heterozygote cell.
// N[All] is not the four-cell total required for this restricted 2x2 test.
__device__ bool ctEpiRel(const float CT[2][3][3], const float N[3], T_Res &RTmp){
	bool found = false;
	float A,B,C,D,Chi2;
	for(int r=0;r<3;r++){
		for(int c=0;c<3;c++){
			if((r==1) && (c==1)){
				continue;
			} else {
				A = CT[Aff][r][c];
				B = CT[Unaff][r][c];
				// confronto sempre rispetto al riferimento: Aa-Bb
				C = CT[Aff][1][1];
				D = CT[Unaff][1][1];
				if((A>0)&&(B>0)&&(C>0)&&(D>0)){
					// Chi Quadro
					float temp = (A*D)-(C*B);
					Chi2 = (temp * temp * N[All]) / ((A+C)*(A+B)*(C+D)*(B+D));
					// Chi quadro corretto Yates
					//float N =A+B+C+D //float N = nCC - NA[Aff] - NA[Unaff] 
					//float Chi2Yates=(pow(fabs((A*D)-(C*B))-(N/2),2)*N)/((A+C)*(A+B)*(C+D)*(B+D));
					if(Chi2 > RTmp.stat){
						found = true;
						RTmp.stat = Chi2;
						RTmp.risk = (A*D)/(B*C); 
					}
				}
			}
		}
	}
	return found;
}

// One thread per SNP pair; M is SNP-major, CC must contain only 0 or 1.
// Negative genotypes exclude a sample from this pair; valid genotypes are 0..2.
// Diagonal tiles reflect the lower triangle; other tiles assume equal halves.
__global__ void VecEpi (TF* M, TF* CC, float *d_Geno, int nSnp, int nSamp, float NAff, float NUnaff, bool diagonalMatrix, unsigned *ResIndex, int NumRes, T_Res *Res, unsigned *NTest)
{
	float CT[2][3][3];
	float NA[2];
	//float p(1);
	float z(0);
	TF* S1 = 0;
	TF* S2 = 0;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	if(diagonalMatrix){
		if(i > j){
			i = nSnp-1-i;
			j = nSnp-1-j;
		}
	} else {
		j += nSnp/2;
	}
	if(i == j) return;

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

		// screening epistasis VS fastepi z-score
		//p = 0.;
		z = fastepi(CT);
	
		// deep epistasis
		// P_Epi is a p-value-style cutoff, but fastepi currently returns a z score.
		if(z < P_Epi){
		//if(z > 0){
			atomicInc(NTest, 10000);
			float N[3];	// corregge il parametro di casi e controlli con il conteggio degli effettivi
			N[Aff]		= NAff-NA[Aff];
			N[Unaff]	= NUnaff-NA[Unaff];
			N[All]		= N[Aff] + N[Unaff];
			bool found(false);
			T_Res RTmp;
			RTmp.p = z;
			RTmp.stat = 0;
			found = ctEpiAbs(CT,N,RTmp);
			// check VS univariate score(Geno or Allelic)
			if(found && (RTmp.stat > d_Geno[i]) && (RTmp.stat > d_Geno[j])){
				RTmp.s1 = i;
				RTmp.s2 = j;
				// pos 0 = Err. | pos [1-NR] = valid | initially RIndex[0] = 1;
				unsigned pos = atomicInc(ResIndex, NumRes);
				if(pos > 0){
					Res[pos] = RTmp;
				}
			}
		}
	}
}

// Allocate and upload one tile, launch its pair scan, then copy results back.
// Caller releases device allocations through CudaCleanup after each tile.
// Known limitations: floor-divided grids, uninitialized d_R, wrapping counters.
void kernelLaunch(TF *h_M, int Msize, TF *h_CC, int Csize, float *h_Geno, int Gsize, T_Res *h_R, int Rsize, int nSnp, int nSamp, float NAff, float NUnaff, bool diagonalMatrix, unsigned *h_NTest){

	// Allocate vectors in device memory
	cutilSafeCall( cudaMalloc((void**)&d_M, Msize) );
	cutilSafeCall( cudaMalloc((void**)&d_Geno, Gsize) );
	cutilSafeCall( cudaMalloc((void**)&d_CC, Csize) );
	cutilSafeCall( cudaMalloc((void**)&d_R, Rsize) );
	cutilSafeCall( cudaMalloc((void**)&d_RInd, sizeof(unsigned)) );
	cutilSafeCall( cudaMalloc((void**)&d_NTest, sizeof(unsigned)) );

    // Copy vectors from host memory to device memory
	cutilSafeCall( cudaMemcpy(d_M, h_M, Msize, cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(d_Geno, h_Geno, Gsize, cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(d_CC, h_CC, Csize, cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(d_RInd, &initval, sizeof(unsigned), cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(d_NTest,&initval, sizeof(unsigned), cudaMemcpyHostToDevice) );

    // Invoke kernel
	dim3 threadsPerBlock(16, 16);
	if(diagonalMatrix){
		// nsnps * nsnps
		// griglia N/2 x N --> matrice triangolare: sottomatrici sulla diagonale
		dim3 blocksPerGrid( (nSnp/2)/threadsPerBlock.x, nSnp/threadsPerBlock.y, 1);
		VecEpi<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_CC, d_Geno, nSnp, nSamp, NAff, NUnaff, diagonalMatrix, d_RInd, (Rsize/sizeof(T_Res))-1, d_R, d_NTest);
	} else {
		// griglia N x N --> matrice completa
		dim3 blocksPerGrid( (nSnp/2)/threadsPerBlock.x, (nSnp/2)/threadsPerBlock.y, 1);
		VecEpi<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_CC, d_Geno, nSnp, nSamp, NAff, NUnaff, diagonalMatrix, d_RInd, (Rsize/sizeof(T_Res))-1, d_R, d_NTest);
	}
	cutilCheckMsg("kernel launch failure");

#ifdef _DEBUG
    cutilSafeCall( cudaThreadSynchronize() );
#endif
	// Copy result from device memory to host memory
	cutilSafeCall( cudaMemcpy(h_R, d_R, Rsize, cudaMemcpyDeviceToHost) );
	cutilSafeCall( cudaMemcpy(h_NTest, d_NTest, sizeof(unsigned), cudaMemcpyDeviceToHost) );
}

void CudaInfo(){
	cudaDeviceProp prop;
	int dev, count;
	// Select best GPU
	cudaSetDevice( cutGetMaxGflopsDeviceId() );
	// Query selected GPU
	cutilSafeCall( cudaGetDeviceCount( &count ) );
	cutilSafeCall( cudaGetDevice( &dev ) );
	printf( "ID of current CUDA device: %d\n", dev );
	cutilSafeCall( cudaGetDeviceProperties( &prop, dev ) );
	printf( " --- General Information for device %d ---\n", dev );
	printf( "Name: %s\n", prop.name );
	printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
	printf( "Clock rate: %d\n", prop.clockRate );
	printf( "Device copy overlap: " );
	if (prop.deviceOverlap)
		printf( "Enabled\n" );
	else
		printf( "Disabled\n" );
	printf( "Kernel execition timeout : " );
	if (prop.kernelExecTimeoutEnabled)
		printf( "Enabled\n" );
	else
		printf( "Disabled\n" );
	printf( " --- Memory Information for device %d ---\n", dev );
	printf( "Total global mem: %ld\n", prop.totalGlobalMem );
	printf( "Total constant Mem: %ld\n", prop.totalConstMem );
	printf( "Max mem pitch: %ld\n", prop.memPitch );
	printf( "Texture Alignment: %ld\n", prop.textureAlignment );
	printf( " --- MP Information for device %d ---\n", dev );
	printf( "Multiprocessor count: %d\n", prop.multiProcessorCount );
	printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
	printf( "Registers per mp: %d\n", prop.regsPerBlock );
	printf( "Threads in warp: %d\n", prop.warpSize );
	printf( "Max threads per block: %d\n", prop.maxThreadsPerBlock );
	printf( "Max thread dimensions: (%d, %d, %d)\n",
		prop.maxThreadsDim[0], prop.maxThreadsDim[1],
		prop.maxThreadsDim[2] );
	printf( "Max grid dimensions: (%d, %d, %d)\n",
		prop.maxGridSize[0], prop.maxGridSize[1],
		prop.maxGridSize[2] );
	printf( "\n" );
}

void CudaCleanup(void){

    // Free device memory
	if (d_M){
        cudaFree(d_M);
		d_M = 0;
	}
	if (d_Geno){
        cudaFree(d_Geno);
		d_Geno = 0;
	}
	if (d_CC){
        cudaFree(d_CC);
		d_CC = 0;
	}
	if (d_R){
        cudaFree(d_R);
		d_R = 0;
	}
	if (d_RInd){
        cudaFree(d_RInd);
		d_RInd = 0;
	}
        
    cutilSafeCall( cudaThreadExit() );
}