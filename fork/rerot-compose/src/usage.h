#pragma once

#include <ostream>

namespace tq::rerot_compose {

inline void usage(std::ostream & out) {
    out <<
        "Usage: tq-rerot-compose --model MODEL.gguf --prompt TEXT\n"
        "                        --stage NAME=INSTRUCTION [options]\n"
        "\n"
        "Run an explicit task graph on one shared KV whiteboard. Each bound\n"
        "stage advances one body token per frontier. Its peers read that token\n"
        "on their next forward step. Completed work remains visible.\n"
        "The assistant-eot profile uses complete native assistant stages.\n"
        "\n"
        "  --stage NAME=TEXT      Add a worker; repeat in the desired plan order.\n"
        "  --after CHILD=PARENT   Require a parent to finish before child starts.\n"
        "                        Repeat for multiple edges. Peers still share KV.\n"
        "  --synthesis TEXT       Final instruction after all workers finish.\n"
        "  --parallel N           Resident worker limit; default: 4. Excess ready\n"
        "                        workers wait for a bound worker to finish.\n"
        "  --worker-tokens N      Body token limit per worker; default: 256.\n"
        "  --final-tokens N       Body token limit for synthesis; default: 256.\n"
        "                        Reaching a limit without source end reports an\n"
        "                        incomplete episode; it does not invent an end.\n"
        "  --ctx-size N           Logical reader context; default: 8192.\n"
        "  --total-kv N           Physical shared token capacity; default: ctx-size.\n"
        "  --batch-size N         Rows per A call; default: 512.\n"
        "  --ubatch-size N        A microbatch limit; default: 128.\n"
        "  --threads N            CPU threads; default: 4.\n"
        "  --gpu-layers N         Layers requested on GPU; default: 0.\n"
        "  --devices NAME,...     Explicit GGML devices, in split order.\n"
        "  --split-mode MODE      none, layer, or row; default: layer.\n"
        "  --chat-template NAME   Override the model's supported native template.\n"
        "  --seed N               Stage-local sampler seed in [0,4294967294].\n"
        "  --temperature X        Non-negative temperature; default: 0 (greedy).\n"
        "  --top-k N              Shortlist for stochastic sampling; default: 40.\n"
        "  --no-phase-memory      Keep transient compute memory between phases.\n"
        "  --phase-memory         Release it before body generation (default).\n"
        "  --json                 Flush config, frontier, token, and result events.\n"
        "  --help                 Show this help without loading a model.\n"
        "\n"
        "The current provider calls installed A and reuses TQRelay submission\n"
        "accounting. It does not provide the whole-model TP5, LateBind, or MTP\n"
        "execution modes. Explicit requests for those modes are rejected.\n"
        "\n"
        "Example:\n"
        "  tq-rerot-compose --model model.gguf --prompt 'Compare these designs.' \\\n"
        "    --stage cost='Assess the cost in one short paragraph.' \\\n"
        "    --stage risk='Assess the risks in one short paragraph.' --parallel 2\n";
}

} // namespace tq::rerot_compose
