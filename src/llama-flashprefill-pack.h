#pragma once

// Internal host-side graph packing. Exposed here so regression tests run
// the actual descriptor conversion, not a separately reimplemented oracle.
#include "llama.h"
#include "llama-flashprefill.h"

#include <cstdint>
#include <string>
#include <vector>

struct llama_flashprefill_layout;

struct llm_fp_rowrec {
    int32_t domain  = -1;
    int32_t src_q   = 0;
    int32_t kv_head = 0;
    int32_t q_head  = 0;
    int32_t tile    = 0;
    int32_t log_pos = 0;
    int32_t pbegin  = 0;
    int32_t pend    = 0;
    bool forced    = false;
};

struct llm_fp_userec {
    int32_t domain    = -1;
    int32_t frag      = 0;
    int32_t tile      = 0;
    int32_t kv_head   = 0;
    int32_t q_group   = 0;
    int64_t sub_off   = 0; // absolute offset in the packed cell table
    int64_t sub_count = 0;
    int32_t flags     = 0;
    int32_t src_q     = 0;
};

struct llm_fp_pack {
    std::vector<llm_fp_rowrec> rows;
    std::vector<llm_fp_userec> uses;
    std::vector<int32_t> cells;
    std::vector<uint32_t> frag_base;
    int64_t n_tiles = 0;
    int64_t max_sel = 0;
    int32_t n_groups = 0;
    std::vector<int32_t> qpos; // phase positions for RERoT only
    int32_t sparse_rows = 0;
    int32_t forced_rows = 0;
    uint64_t visible_tokens = 0;
};

enum llm_fp_pack_rc {
    LL_FP_PACK_OK       = 0,
    LL_FP_PACK_CORRUPT  = 1,
    LL_FP_PACK_OVERFLOW = 2,
};

LLAMA_API llm_fp_pack_rc llm_fp_pack_live(
    const llama_flashprefill_layout & layout,
    const std::vector<llama_flashprefill_row> & rows,
    const llama_flashprefill_config & cfg,
    uint32_t hq, uint32_t hkv, uint32_t gqa, uint32_t n_tokens,
    llm_fp_pack * out, std::string * error, bool dry_run = false);
