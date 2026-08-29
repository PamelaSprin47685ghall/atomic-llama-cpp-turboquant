#include "calib-dump.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace wxq {

bool load_calib(const std::string & path, const std::vector<int> & want, bool want_cov,
                calib_dump & out, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }

    char magic[8] = {};
    f.read(magic, 8);
    if (std::memcmp(magic, "WXQCAL03", 8) != 0) {
        err = "bad magic in " + path + " (not a WXQCAL03 dump)";
        return false;
    }

    calib_meta & m = out.meta;
    uint8_t pad[7] = {};
    f.read((char *) &m.n_entries,    sizeof(m.n_entries));
    f.read((char *) &m.max_ctx,      sizeof(m.max_ctx));
    f.read((char *) &m.act_stride,   sizeof(m.act_stride));
    f.read((char *) &m.act_tokens,   sizeof(m.act_tokens));
    f.read((char *) &m.route_stride, sizeof(m.route_stride));
    f.read((char *) &m.flags,        sizeof(m.flags));
    f.read((char *) &m.n_decoded,    sizeof(m.n_decoded));
    f.read((char *) &m.n_total,      sizeof(m.n_total));
    f.read((char *) &m.n_ctx_resets, sizeof(m.n_ctx_resets));
    f.read((char *) &m.complete,     sizeof(m.complete));
    f.read((char *) pad,             sizeof(pad));
    if (!f) {
        err = "truncated header in " + path;
        return false;
    }

    for (uint32_t i = 0; i < m.n_entries; ++i) {
        uint32_t layer = 0, n_expert = 0, n_embd = 0, n_ff = 0;
        uint64_t n_pos = 0, n_route = 0, n_cov = 0;
        f.read((char *) &layer,    sizeof(layer));
        f.read((char *) &n_expert, sizeof(n_expert));
        f.read((char *) &n_embd,   sizeof(n_embd));
        f.read((char *) &n_ff,     sizeof(n_ff));
        f.read((char *) &n_pos,    sizeof(n_pos));
        f.read((char *) &n_route,  sizeof(n_route));
        f.read((char *) &n_cov,    sizeof(n_cov));
        if (!f) {
            err = "truncated entry " + std::to_string(i) + " in " + path;
            return false;
        }

        const bool keep = want.empty() ||
            std::find(want.begin(), want.end(), (int) layer) != want.end();

        calib_layer  scratch;
        calib_layer & e = keep ? out.layers[(int) layer] : scratch;
        e.n_expert = n_expert;
        e.n_embd   = n_embd;
        e.n_ff     = n_ff;
        e.n_pos    = n_pos;
        e.n_route  = n_route;
        e.n_cov    = n_cov;

        auto rd_u64 = [&](std::vector<uint64_t> & v, size_t n) {
            v.resize(n);
            f.read((char *) v.data(), (std::streamsize) (n * sizeof(uint64_t)));
        };
        auto rd_f64 = [&](std::vector<double> & v, size_t n) {
            v.resize(n);
            f.read((char *) v.data(), (std::streamsize) (n * sizeof(double)));
        };

        rd_u64(e.counts,  n_expert);
        rd_f64(e.act_sum, n_embd);
        if (m.flags & CALIB_EXPERT_IN) {
            rd_u64(e.exp_n,    n_expert);
            rd_f64(e.exp_mean, (size_t) n_expert * n_embd);
            rd_f64(e.exp_diag, (size_t) n_expert * n_embd);
        }
        if (m.flags & CALIB_HIDDEN) {
            rd_u64(e.hid_n,   n_expert);
            rd_f64(e.hid_sum, (size_t) n_expert * n_ff);
        }
        if (m.flags & CALIB_COV) {
            if (keep && want_cov) {
                rd_f64(e.cov, (size_t) n_embd * n_embd);
            } else {
                f.seekg((std::streamoff) ((size_t) n_embd * n_embd * sizeof(double)), std::ios::cur);
            }
        }
        if (!f) {
            err = "truncated payload for layer " + std::to_string(layer) + " in " + path;
            return false;
        }
        if (!keep) {
            scratch = calib_layer();
        }
    }

    return true;
}

bool write_calib(const std::string & path, const calib_meta & meta,
                 const std::map<int, calib_layer> & layers, std::string & err) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            err = "cannot write " + tmp;
            return false;
        }

        out.write("WXQCAL03", 8);
        const uint32_t n_entries = (uint32_t) layers.size();
        out.write((const char *) &n_entries,         sizeof(n_entries));
        out.write((const char *) &meta.max_ctx,      sizeof(meta.max_ctx));
        out.write((const char *) &meta.act_stride,   sizeof(meta.act_stride));
        out.write((const char *) &meta.act_tokens,   sizeof(meta.act_tokens));
        out.write((const char *) &meta.route_stride, sizeof(meta.route_stride));
        out.write((const char *) &meta.flags,        sizeof(meta.flags));
        out.write((const char *) &meta.n_decoded,    sizeof(meta.n_decoded));
        out.write((const char *) &meta.n_total,      sizeof(meta.n_total));
        out.write((const char *) &meta.n_ctx_resets, sizeof(meta.n_ctx_resets));
        out.write((const char *) &meta.complete,     sizeof(meta.complete));
        const char pad[7] = {};
        out.write(pad, sizeof(pad));

        std::vector<double> row;  // mirrored cov row scratch
        for (const auto & [il, e] : layers) {
            const uint32_t layer = (uint32_t) il;
            out.write((const char *) &layer,      sizeof(layer));
            out.write((const char *) &e.n_expert, sizeof(e.n_expert));
            out.write((const char *) &e.n_embd,   sizeof(e.n_embd));
            out.write((const char *) &e.n_ff,     sizeof(e.n_ff));
            out.write((const char *) &e.n_pos,    sizeof(e.n_pos));
            out.write((const char *) &e.n_route,  sizeof(e.n_route));
            out.write((const char *) &e.n_cov,    sizeof(e.n_cov));
            out.write((const char *) e.counts.data(),  (std::streamsize) (e.counts.size()  * sizeof(uint64_t)));
            out.write((const char *) e.act_sum.data(), (std::streamsize) (e.act_sum.size() * sizeof(double)));

            if ((meta.flags & CALIB_EXPERT_IN) && !e.exp_mean.empty()) {
                out.write((const char *) e.exp_n.data(),    (std::streamsize) (e.exp_n.size()    * sizeof(uint64_t)));
                out.write((const char *) e.exp_mean.data(), (std::streamsize) (e.exp_mean.size() * sizeof(double)));
                out.write((const char *) e.exp_diag.data(), (std::streamsize) (e.exp_diag.size() * sizeof(double)));
            }
            if ((meta.flags & CALIB_HIDDEN) && !e.hid_sum.empty()) {
                out.write((const char *) e.hid_n.data(),   (std::streamsize) (e.hid_n.size()   * sizeof(uint64_t)));
                out.write((const char *) e.hid_sum.data(), (std::streamsize) (e.hid_sum.size() * sizeof(double)));
            }
            if ((meta.flags & CALIB_COV) && !e.cov.empty()) {
                // only the upper triangle is live in the collector; mirror row by
                // row so the consumer gets a plain square matrix
                const size_t d = e.n_embd;
                row.resize(d);
                for (size_t i = 0; i < d; ++i) {
                    for (size_t j = 0; j < d; ++j) {
                        row[j] = (j >= i) ? e.cov[i*d + j] : e.cov[j*d + i];
                    }
                    out.write((const char *) row.data(), (std::streamsize) (d * sizeof(double)));
                }
            }
        }
        if (!out) {
            err = "short write to " + tmp;
            return false;
        }
    }

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        err = "cannot rename " + tmp + " -> " + path + ": " + ec.message();
        return false;
    }
    return true;
}

} // namespace wxq
