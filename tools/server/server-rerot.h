#pragma once

#include "llama-rerot.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Server-side control-plane helpers for Recursive Elastic Ring-of-Thought.
//
// The model-facing planner protocol is byte-oriented, while KV visibility is
// token/cell-oriented. The parser therefore deliberately classifies an entire
// sampled token conservatively: if any suffix could become the structural
// `<ol>` opener, that token is PENDING until the prefix is resolved. This may
// temporarily hide harmless bytes in the same token from foreign readers, but
// it never exposes an incomplete structural record.

enum class server_rerot_parser_state : uint8_t {
    public_text = 0,
    opening_candidate,
    list_pending,
    complete,
    failed,
};

struct server_rerot_parser_step {
    // Visibility to use when writing the sampled token that supplied this byte
    // chunk. PRIVATE planner injection is handled by the runtime before model
    // sampling and therefore never originates from this parser.
    llama_rerot_visibility write_visibility = llama_rerot_visibility::public_live;

    // A PENDING opener candidate from one or more previous tokens was proven
    // not to be `<ol>`. The runtime must atomically reclassify the previous
    // pending run to PUBLIC before installing the write tag for this token.
    bool release_previous_pending = false;

    // Structural events are committed only after the current token has been
    // decoded, so its KV cell participates in the same atomic publication.
    bool record_opened = false;
    bool record_closed = false;
    bool malformed = false;

    // Populated only when record_closed is true. Titles are trimmed at their
    // outer ASCII whitespace but otherwise preserved byte-for-byte.
    std::vector<std::string> items;
    std::string error;
};

class server_rerot_planner_parser {
public:
    server_rerot_planner_parser();

    void reset();

    // Consume exactly the bytes represented by one sampled token. Token
    // boundaries may occur anywhere inside `<ol>`, `<li>`, `</li>`, or `</ol>`.
    server_rerot_parser_step consume(std::string_view bytes);

    server_rerot_parser_state state() const;
    bool complete() const;
    bool failed() const;

    const std::vector<std::string> & items() const;
    const std::string & error() const;

private:
    server_rerot_parser_step consume_before_list(std::string_view bytes);
    server_rerot_parser_step consume_list_bytes(std::string_view bytes, bool opened_now);
    bool finish_record(server_rerot_parser_step & step);
    bool fail(server_rerot_parser_step & step, std::string message);

    server_rerot_parser_state state_ = server_rerot_parser_state::public_text;

    // At most "<ol". These bytes live in one pending run in KV until a later
    // token either completes `<ol>` or disproves the candidate.
    std::string opener_candidate_;

    // Begins exactly at the structural `<ol>` opener. Bytes preceding the
    // opener in the same tokenizer token are conservatively PENDING in KV but
    // are intentionally not part of the semantic list record parsed here.
    std::string list_bytes_;

    std::vector<std::string> items_;
    std::string error_;
};

