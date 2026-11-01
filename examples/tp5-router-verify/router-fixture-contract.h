#pragma once

#include <string>
#include <vector>

// Contract check for the router-scratch fixture: every metadata key the real
// Qwen4Exp loader reads must exist in the generated file with the right type,
// length and value, and nothing the model does not ask for may be silently
// missing. The reference is src/models/qwen4exp.cpp load_arch_hparams itself,
// exercised through the real llama_model_loader — not a copy of the key list.
//
// Needs the in-tree loader (src/), so this unit is only built when the fixture
// is configured inside this source tree (see CMakeLists.txt); the plain
// binary-only build skips it and the generator still works.

struct fixture_contract_diff {
    std::vector<std::string> problems; // empty == the loader accepts the file
    int n_meta_checked    = 0;
    int n_tensors_checked = 0;
};

// Runs load_arch_hparams (plus the scalar/array checks that failure would not
// localise) and load_arch_tensors against model_path. Throws only when the file
// cannot be opened as GGUF.
fixture_contract_diff fixture_check_contract(const std::string & model_path);
