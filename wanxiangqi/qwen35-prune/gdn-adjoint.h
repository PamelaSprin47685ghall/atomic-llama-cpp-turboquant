#pragma once

#include <vector>

namespace qwen35_prune {

struct gdn_scalar_gate_forward {
    int n_tokens = 0;
    int dim = 0;
    std::vector<double> output;
    std::vector<double> final_state;
    // states[t] is S_t before token t, states[n_tokens] is the final state.
    std::vector<double> states;
};

struct gdn_scalar_gate_grad {
    std::vector<double> q;
    std::vector<double> k;
    std::vector<double> v;
    std::vector<double> g;
    std::vector<double> beta;
    std::vector<double> initial_state;
};

// Reference scalar-gate Gated DeltaNet recurrence for one head and one
// sequence. Layout is row-major [token, dim] for q/k/v and [dim, dim] for
// states. This intentionally mirrors GGML_OP_GATED_DELTA_NET when g.ne[0] == 1.
gdn_scalar_gate_forward gdn_scalar_gate_forward_ref(
        const std::vector<double> & q,
        const std::vector<double> & k,
        const std::vector<double> & v,
        const std::vector<double> & g,
        const std::vector<double> & beta,
        const std::vector<double> & initial_state,
        int n_tokens,
        int dim);

// Exact reverse-mode derivative of the reference recurrence. d_output is
// dL/dy_t and d_final_state is dL/dS_T. Complexity is O(T * dim^2), the same
// asymptotic complexity as the forward recurrence.
gdn_scalar_gate_grad gdn_scalar_gate_backward_ref(
        const std::vector<double> & q,
        const std::vector<double> & k,
        const std::vector<double> & v,
        const std::vector<double> & g,
        const std::vector<double> & beta,
        const std::vector<double> & initial_state,
        const std::vector<double> & d_output,
        const std::vector<double> & d_final_state,
        int n_tokens,
        int dim);

} // namespace qwen35_prune
