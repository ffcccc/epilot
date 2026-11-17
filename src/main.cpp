#include "include/bed_dataset.h"
#include "include/epi_cpu.h"
#include "include/epi_gpu.cuh"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::vector<std::string> files;
    std::vector<std::string> methods{"fast"};
    std::string device = "cpu";
    bool verbose = false;
    bool prompt = true;
    bool help = false;
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " --files <a.bed> [b.bed ...] [--methods fast chi2 ...]"
                 " [--device cpu|gpu] [--verbose] [--noprompt]\n"
                 "Methods: fast (fastepistasis+Chi2), chi2, fastepistasis,"
                 " fastepistasis+Chi2\n";
}

bool parse_arguments(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else if (arg == "--files" || arg == "--methods") {
            std::vector<std::string>* values =
                arg == "--files" ? &options.files : &options.methods;
            if (arg == "--methods") values->clear();
            const int first = i + 1;
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
                values->emplace_back(argv[++i]);
            if (static_cast<int>(values->size()) == 0 ||
                (arg == "--methods" && i + 1 == first)) {
                std::cerr << "error: " << arg << " requires at least one value\n";
                return false;
            }
        } else if (arg == "--device") {
            if (i + 1 >= argc) {
                std::cerr << "error: --device requires cpu or gpu\n";
                return false;
            }
            options.device = argv[++i];
        } else if (arg == "--verbose") {
            options.verbose = true;
        } else if (arg == "--noprompt") {
            options.prompt = false;
        } else {
            std::cerr << "error: unexpected argument '" << arg << "'\n";
            return false;
        }
    }

    if (options.help) return true;
    if (options.files.empty()) {
        std::cerr << "error: --files requires at least one BED file\n";
        return false;
    }
    if (options.device != "cpu" && options.device != "gpu") {
        std::cerr << "error: --device must be cpu or gpu\n";
        return false;
    }
    return true;
}

std::string normalize_method(const std::string& method) {
    if (method == "fast") return "fastepistasis+Chi2";
    if (method == "chi2") return "Chi2";
    if (method == "fastepistasis" || method == "fastepistasis+Chi2" ||
        method == "Chi2")
        return method;
    throw std::invalid_argument("unsupported method '" + method +
                                "' (available: fast, chi2, fastepistasis, "
                                "fastepistasis+Chi2)");
}

void append_results(const std::string& path, const std::string& method,
                    const BedDataset& source, const std::vector<epi::Hit>& hits) {
    std::ofstream out(path + ".cc.log", std::ios_base::out | std::ios_base::app);
    if (!out) throw std::runtime_error("cannot open output file " + path + ".cc.log");

    out << "Test\tEdgeName\tCHR1\tSNP1\tCHR2\tSNP2\tSTAT\tOR\tSource\tTarget\n";
    for (size_t i = 0; i < hits.size(); ++i) {
        const epi::Hit& hit = hits[i];
        std::string chr1, chr2, name1, name2;
        source.getSNPChr(hit.s1, chr1);
        source.getSNPName(hit.s1, name1);
        source.getSNPChr(hit.s2, chr2);
        source.getSNPName(hit.s2, name2);
        out << method << "\tR" << i << '\t' << chr1 << '\t' << name1 << '\t'
            << chr2 << '\t' << name2 << '\t' << hit.stat << '\t' << hit.risk
            << '\t' << hit.s1 << '\t' << hit.s2 << '\n';
    }
    if (!out) throw std::runtime_error("failed while writing output file " + path + ".cc.log");
}

int run(const Options& options) {
    for (const std::string& path : options.files) {
        try {
            BedDataset source(path);
            epi::ScanConfig config;
            for (const std::string& requested_method : options.methods) {
                const std::string method = normalize_method(requested_method);
                config.test = method;
                const epi::ScanContext context = epi::make_context(source, config);
                epi::ScanStats stats;
                std::vector<epi::Hit> hits;

                if (options.device == "gpu") {
                    epi::GpuTileScanner scanner(context, config);
                    hits = epi::run_scan(source, context, scanner, config, &stats);
                } else {
                    epi::CpuTileScanner scanner(context, config);
                    hits = epi::run_scan(source, context, scanner, config, &stats);
                }

                append_results(path, method, source, hits);
                std::cout << path << ": " << method << " on " << options.device
                          << " scanned " << stats.n_tiles << " tiles; found "
                          << stats.n_hits << " hits\n";
                if (options.verbose)
                    std::cout << "  tested " << stats.n_tests << " SNP pairs\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "error: " << path << ": " << e.what() << '\n';
            return 1;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_arguments(argc, argv, options)) {
        print_usage(argv[0]);
        return 1;
    }
    if (options.help) {
        print_usage(argv[0]);
        return 0;
    }

    const int result = run(options);
    if (result != 0) return result;
    if (options.prompt) {
        std::printf("\nPress ENTER to exit...\n");
        std::fflush(stdout);
        std::getchar();
    }
    return 0;
}
