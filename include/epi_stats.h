// Epistatic tests. Each test is a struct with one static function, usable on host and
// device, so CPU and GPU run the same code.
//
//   static EPI_HD TestOut run(const real CT[2][3][3], const real N[3], const TestOptions&);
//
// CT[g][a][b]: samples in group g (UNAFF/AFF) with SNP1 genotype a and SNP2 genotype b.
// N[AFF], N[UNAFF]: samples with both genotypes known; N[ALL] = N[AFF] + N[UNAFF].
//
// To add a test: write a struct with run() and add one line to with_test().
#ifndef EPI_TESTS_H
#define EPI_TESTS_H
#include "epi_core.h"
#include <math.h>
#include <stdexcept>
#include <string>
namespace epi {

/// Fast-epistasis statistic (PLINK 1.07 --fast-epistasis compatible).
// z = |log OR_case - log OR_ctrl| / sqrt(v_case + v_ctrl)
// Returns out.stat = z*z  (chi-square, 1 df). Two-sided p = 2*Phi(-|z|).
// Returns stat = 0 if any allele-pair cell is empty (PLINK drops such pairs).
struct FastEpistasis {
    static EPI_HD TestOut run(const real CT[2][3][3], const real N[3], const TestOptions&) {
        (void)N;
        TestOut out = {0, 0, -1, -1};
        real T[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};  // per group: A, B, C, D
        for (int g = 0; g < 2; ++g)
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) {
                    const real n = CT[g][a][b];
                    T[g][0] += n * (2 - a) * (2 - b);  // allele 0 at SNP1, allele 0 at SNP2
                    T[g][1] += n * (2 - a) * b;        // allele 0, allele 1
                    T[g][2] += n * a * (2 - b);        // allele 1, allele 0
                    T[g][3] += n * a * b;              // allele 1, allele 1
                }
        real var = 0, logOR[2] = {0, 0};
        for (int g = 0; g < 2; ++g) {
            const real A = T[g][0], B = T[g][1], C = T[g][2], D = T[g][3];
            if (!(A > 0 && B > 0 && C > 0 && D > 0)) return out;
            var += 1 / A + 1 / B + 1 / C + 1 / D;
            logOR[g] = log(A * D) - log(B * C);
        }
        real z = fabs(logOR[AFF] - logOR[UNAFF]) / sqrt(var);
        out.stat = z*z;
        return out;
    }
};

struct FastEpistasis2 {
    static EPI_HD TestOut run(const real CT[2][3][3], const real N[3], const TestOptions&){
        //float p(1);
        TestOut out = {0, 0, -1, -1};

        real A = 4*CT[AFF][0][0]	+ 2*CT[AFF][0][1]	+ 2*CT[AFF][1][0]	+ CT[AFF][1][1];
        real B = 4*CT[AFF][0][2]	+ 2*CT[AFF][0][1]	+ 2*CT[AFF][1][2]	+ CT[AFF][1][1];
        real C = 4*CT[AFF][2][0]	+ 2*CT[AFF][2][1]	+ 2*CT[AFF][1][0]	+ CT[AFF][1][1];
        real D = 4*CT[AFF][2][2]	+ 2*CT[AFF][2][1]	+ 2*CT[AFF][1][2]	+ CT[AFF][1][1];

        real a = 4*CT[UNAFF][0][0]	+ 2*CT[UNAFF][0][1]	+ 2*CT[UNAFF][1][0]	+ CT[UNAFF][1][1];
        real b = 4*CT[UNAFF][0][2]	+ 2*CT[UNAFF][0][1]	+ 2*CT[UNAFF][1][2]	+ CT[UNAFF][1][1];
        real c = 4*CT[UNAFF][2][0]	+ 2*CT[UNAFF][2][1]	+ 2*CT[UNAFF][1][0]	+ CT[UNAFF][1][1];
        real d = 4*CT[UNAFF][2][2]	+ 2*CT[UNAFF][2][1]	+ 2*CT[UNAFF][1][2]	+ CT[UNAFF][1][1];
        
        if((A>0)&&(a>0)&&(B>0)&&(b>0)&&(C>0)&&(c>0)&&(D>0)&&(d>0)){
            real varCa = 1.0/A + 1.0/D + 1.0/B + 1.0/C;
            real varCo = 1.0/a + 1.0/d + 1.0/b + 1.0/c;
            real orCa	= (A*D)/(B*C);
            real orCo	= (a*d)/(b*c);
            real z = fabs( (log(orCa) - log(orCo)) / sqrt(varCa + varCo) );
            out.stat = z*z;
        }
        return out;
    }
};


// Pearson chi-square for every joint-genotype cell against the rest (2x2: cell vs. rest,
// cases vs. controls). Reports the cell with the largest chi-square; ties keep the first
// cell in row-major order. Cells with a zero count in either group are skipped.
struct Chi2Absolute {
    static EPI_HD TestOut run(const real CT[2][3][3], const real N[3], const TestOptions&) {
        TestOut best = {0, 0, -1, -1};
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                const real A = CT[AFF][r][c];     // cases in cell
                const real B = CT[UNAFF][r][c];   // controls in cell
                const real C = N[AFF] - A;        // cases outside cell
                const real D = N[UNAFF] - B;      // controls outside cell
                if (!(A > 0 && B > 0 && C > 0 && D > 0)) continue;
                const real temp = A * D - C * B;
                const real chi2 = temp * temp * N[ALL] / ((A + C) * (A + B) * (C + D) * (B + D));
                if (chi2 > best.stat) {
                    best.stat = chi2;
                    best.risk = (A * D) / (B * C);
                    best.row = r;
                    best.col = c;
                }
            }
        return best;
    }
};

// Two-stage test: FastEpistasis screen (|z| >= opt.screen_min), then Chi2Absolute on
// the pairs that pass. This is the pipeline described in the design notes.
struct FastThenChi2 {
    static EPI_HD TestOut run(const real CT[2][3][3], const real N[3], const TestOptions& opt) {
        const TestOut screen = FastEpistasis2::run(CT, N, opt);
        if (!(screen.stat >= opt.screen_min)) return TestOut{0, 0, -1, -1};
        return Chi2Absolute::run(CT, N, opt);
    }
};

// Host-only: picks the test by name and calls f with an instance of it. f is a generic
// lambda, so each back end can instantiate its kernels per test.
// Names: "fastepistasis", "Chi2", "fastepistasis+Chi2".
template <class F>
auto with_test(const std::string& name, F&& f) -> decltype(f(FastEpistasis())) {
    if (name == "fastepistasis")        return f(FastEpistasis());
    if (name == "Chi2")                 return f(Chi2Absolute());
    if (name == "fastepistasis+Chi2")   return f(FastThenChi2());
    // if (name == "XYZ")   return f( XYZ() );
    throw std::invalid_argument("unknown epistatic test: '" + name + "'");
}

}  // namespace epi
#endif  // EPI_TESTS_H