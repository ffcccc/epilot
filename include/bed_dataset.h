#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "3rdparty/bed-reader-cpp/0.1.0/include/bed_reader_bridge.h"
#include "include/common.h"

// PLINK .bed/.bim/.fam dataset exposing the subset of the hdfTPED interface used by main.cpp.
// Genotypes come from BedWrapper; SNP and sample metadata are parsed from the .bim/.fam siblings.
class BedDataset {
public:
	explicit BedDataset(const std::string &bedPath) : m_filename(bedPath) {
		const std::string base = stripExtension(bedPath);

		::rust::Box<::BedWrapper> bed = new_bed(::rust::Str(bedPath));
		m_nSamples = static_cast<unsigned>(bed->get_iid_count());
		m_nSnps = static_cast<unsigned>(bed->get_sid_count());

		readBim(base + ".bim");
		readFam(base + ".fam");

		// bed-reader returns an (iid x sid) Fortran-order matrix: each SNP's samples are contiguous.
		::rust::Vec<float> values = bed->read_all_genotypes();
		const size_t total = static_cast<size_t>(m_nSnps) * m_nSamples;
		if (values.size() != total)
			throw std::runtime_error("unexpected genotype matrix size in " + bedPath);
		m_geno.resize(total);
		for (size_t k = 0; k < total; ++k) {
			const float v = values[k];
			m_geno[k] = std::isnan(v) ? T_GType(-1) : static_cast<T_GType>(v);
		}

		computeSnpScores();
	}

	std::string filename() const { return m_filename; }
	unsigned numSnps() const { return m_nSnps; }
	unsigned numSamples() const { return m_nSamples; }

	void getSNPChr(const unsigned snpInd, std::string &chr) const { chr = m_snpChr.at(snpInd); }
	void getSNPName(const unsigned snpInd, std::string &name) const { name = m_snpName.at(snpInd); }
	void getSNPScore(const unsigned snpInd, float &score) const { score = m_snpScore.at(snpInd); }

	// 1 = affected, 0 = unaffected, -1 = missing (PLINK: 2 = case, 1 = control).
	void getSamplePheno(const unsigned sampInd, int &phe) const { phe = m_pheno.at(sampInd); }

	void getSnpMem(const unsigned snpInd, T_GType *val) const {
		const T_GType *src = &m_geno.at(static_cast<size_t>(snpInd) * m_nSamples);
		std::copy(src, src + m_nSamples, val);
	}

private:
	std::string m_filename;
	unsigned m_nSamples = 0;
	unsigned m_nSnps = 0;
	std::vector<T_GType> m_geno;
	std::vector<std::string> m_snpChr;
	std::vector<std::string> m_snpName;
	std::vector<float> m_snpScore;
	std::vector<int> m_pheno;

	static std::string stripExtension(const std::string &path) {
		const size_t dot = path.find_last_of('.');
		const size_t sep = path.find_last_of("/\\");
		if (dot == std::string::npos || (sep != std::string::npos && dot < sep))
			return path;
		return path.substr(0, dot);
	}

	void readBim(const std::string &path) {
		std::ifstream in(path);
		if (!in)
			throw std::runtime_error("cannot open " + path);
		std::string line, chr, name;
		while (std::getline(in, line)) {
			std::istringstream ss(line);
			if (!(ss >> chr >> name))
				continue;
			m_snpChr.push_back(chr);
			m_snpName.push_back(name);
		}
		if (m_snpName.size() != m_nSnps)
			throw std::runtime_error("SNP count mismatch between .bed and " + path);
	}

	void readFam(const std::string &path) {
		std::ifstream in(path);
		if (!in)
			throw std::runtime_error("cannot open " + path);
		std::string line, fid, iid, pid, mid, sex, pheno;
		while (std::getline(in, line)) {
			std::istringstream ss(line);
			if (!(ss >> fid >> iid >> pid >> mid >> sex >> pheno))
				continue;
			m_pheno.push_back(pheno == "2" ? 1 : (pheno == "1" ? 0 : -1));
		}
		if (m_pheno.size() != m_nSamples)
			throw std::runtime_error("sample count mismatch between .bed and " + path);
	}

	// Single-locus counterpart of ctLocEpiAbs: max 2x2 chi-square of (genotype == g) vs case/control.
	void computeSnpScores() {
		m_snpScore.assign(m_nSnps, 0.f);
		for (unsigned s = 0; s < m_nSnps; ++s) {
			float ct[2][3] = {{0, 0, 0}, {0, 0, 0}};
			float n[2] = {0, 0};
			const T_GType *g = &m_geno[static_cast<size_t>(s) * m_nSamples];
			for (unsigned k = 0; k < m_nSamples; ++k) {
				const int phe = m_pheno[k];
				if (phe < 0 || g[k] < 0 || g[k] > 2)
					continue;
				ct[phe][static_cast<int>(g[k])]++;
				n[phe]++;
			}
			float best = 0.f;
			for (int r = 0; r < 3; ++r) {
				const float A = ct[1][r], B = ct[0][r];
				const float C = n[1] - A, D = n[0] - B;
				if (A > 0 && B > 0 && C > 0 && D > 0) {
					const float t = A * D - C * B;
					const float chi2 = (t * t * (n[0] + n[1])) / ((A + C) * (A + B) * (C + D) * (B + D));
					if (chi2 > best)
						best = chi2;
				}
			}
			m_snpScore[s] = best;
		}
	}
};
