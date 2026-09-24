#pragma once

// Thinking wire: the probe splits the model's own thinking into parallel
// questions, written as a flat HTML list.
//
// The server injects a fixed prefix through the first list item:
//
//   My thinking splits here: copies of me think each question below in parallel
//   and their raw thinking all comes back to me; then I decide — tool call or
//   answer. No question may need another's answer.
//   <ul>Looking at this request, the parts that stand alone are: <li>
//
// and the model continues under GBNF until the closing </ul>.
//
// What this wire is NOT, and why the earlier names failed:
//
//   * "goal / sub-goal" made every plan procedural. A goal is something to
//     achieve, so the model answered with actions that wait on each other
//     ("output it in Chinese", "handle the disputed countries").
//   * "answer / section" implied a deliverable being cut into pieces that are
//     later concatenated. Most of the time -- agent coding above all -- nobody
//     is asking for an answer document, and the merge is not a concatenation.
//
// What actually happens: the *thinking* is distributed over sub-questions and
// aspects, every worker's raw reasoning stays visible for the rest of the
// episode and is consumed as-is, and only at the end does the model decide
// whether to call a tool or write the reply. A transformer's thinking has no
// required format, which is exactly what this exploits. So the unit is a
// question to think about -- never a step to perform, never a section to write.
//
// Compared with the Mermaid mindmap wire the flat list also removes the whole
// indentation failure class: no depth to skip, no parent restating its children,
// no outline numbering to drift into. The markup carries the semantics -- every
// `<li class="question">` re-states what the next span of text has to be -- so
// the natural-language instruction stays one sentence.
//
// Question text is deliberately permissive: any byte except '<' and the ASCII
// line/tab controls, so a question can be long, concrete and multilingual. One
// question is one line of arbitrary length; forbidding raw newlines inside an
// item is what keeps the item boundary unambiguous, the canonical re-render
// stable, and one plan on one log line.
//
// Everything is flat by construction: each question becomes one peer worker
// (synthetic root + depth-2 leaves), which is exactly the shape the physical pen
// cohort scheduler wants. No node/leaf/byte budget: the model's context is the
// only bound.

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace server_thinking {

// Ends with '\n' (the injection contract requires it).
//
// Three hard structural rules, each from a measured failure:
//
//   1. Thinking is allowed ONLY in the <ul> head, never between items. The
//      probe has no thinking budget before it starts, and forcing the first
//      token into an item made plan content unstable (per-continent one run,
//      meta-analysis the next, 1739 tokens of deliberation inside a single
//      item that never reached </ul>, then unrelated specimen questions).
//      Free text between items is worse than useless: the model will think
//      between every pair of items and never finish the list.
//   2. </ul> is reachable only after at least one complete item. Otherwise
//      the first '<' the model types can close an empty list.
//   3. Items are bare <li>...</li>. A lazy grammar armed on a self-written
//      <ul class="thinking"> cannot work -- the model does not spontaneously
//      write that -- and class attributes are the first thing it drops.
//
// States the mechanism, not a list of rules: at planning time the become tool's
// description is not yet in play, so without it the model has no idea what the
// list is for and splits by "how would I do this". Once it knows that copies of
// it think the questions in parallel, that every copy's raw thinking returns to
// it, and that the decision comes afterwards, non-overlapping and
// non-sequential questions follow on their own, in any domain. This is also why
// become's description is a single clause: the contract is stated here once
// instead of being restated in every lane's frame.
//
// It no longer has to end on a colon: the bridge into the list is the head
// starter inside <ul>, not this sentence. (When the list started immediately
// after this sentence, ending it on a full stop made the probe emit a
// restatement line -- "The user's question in Chinese: ..." -- as the whole plan.)
//
// The unit is an ASPECT because that is what the model's own unconstrained
// thinking is made of. Baseline capture on a concrete agent-coding task (add a
// /healthz endpoint to tools/server; 41 paragraphs, 12841 chars, RERoT off):
// about 25 paragraphs circle one unknown about the material ("what is a wire in
// this codebase?"), the rest recall struct fields and then cover the four things
// the request asked to produce. Neither "goal/sub-goal", nor "answer/section",
// nor "question" describes those units: they are the things the thinking has to
// cover -- what must be pinned down, and what must be produced. Hence the two
// kinds named explicitly in the last clause.
//
// Each clause maps to a measured failure of the previous wording: the mechanism
// (split / parallel / raw thinking returns / then decide) against procedural
// plans; "no question may need another's answer" against pipeline stages;
// "here I only ask, one line each" against a probe that wrote its conclusions
// inline (1734 tokens of "Question I: ... errors.py has AppError ..."); "each copy
// is me with nothing new to look at" against a probe that produced 15 clarification
// questions for the user ("Do you want i18n?", "Preserve CLI compatibility?"),
// which no lane can answer by thinking.
//
// Both the instruction tail and the starter are written so they cannot be copied
// as items. With the tools block advertised in C0 (which removes the prefix
// rebuild) the probe started echoing whatever noun phrase sat closest to the
// list: "what I must pin down myself and what I must produce" came back as two
// items, and "what can be thought through on its own here" as one. The tail now
// forbids restating the instruction and the starter points at the request, so
// the first item has to be task-specific.
//
// Rejected wordings, each patching a symptom instead of stating the mechanism:
// "What is the goal? Which sub-goals can run in parallel?" (answered its own
// question inside the list), "each self-contained / concrete task" (still
// procedural), "by category, not by stage" (read as categories of work), "one per
// thing the goal covers" (does not generalise), "cut the answer into parts ...
// joined with nothing added" (there is often no answer document to cut).
inline constexpr std::string_view prompt =
    "My thinking splits here: once the aspects are listed, copies of me think them all in parallel "
    "and their raw thinking comes back to me; then I decide — tool call or answer. No aspect may "
    "need another's outcome, and none of them restates this instruction: each one names something "
    "specific in the request, one per line, in words with no punctuation.\n";

// Injected verbatim after the prompt: it opens the list AND the first item, so the
// first sampled token is already inside an aspect. There is deliberately no free
// "head" region between <ul> and the first <li>. A head was tried three ways and
// all three lost: unbounded it never reached the list (min_p masks the exit token
// once prose starts), with an injected reminder at each breath point the lane
// copied the reminder back verbatim dozens of times, and mandatory-but-bounded it
// only produced a junk bridge clause that dragged the aspects into meta-talk. The
// best measured plans all had an empty head, so the protocol now enforces that.
// The starter ends on a colon and a space, not an ellipsis. With "..." the lane
// was still being invited to reflect, and a forced <li> right after it produced
// the vaguest possible aspects ("what I need to decide now", "what I must give at
// the end"); a colon is the canonical "a list follows" signal, so the injected
// item tag is coherent with the text before it.
//
// Items are bare <li> on purpose -- the model will not reliably produce class
// attributes, and a class it drops is a masked token it cannot recover from.
inline constexpr std::string_view probe_prefix =
    "<ul>Looking at this request, the parts that stand alone are: <li>";

inline constexpr std::string_view open_item  = "<li>";
inline constexpr std::string_view close_item = "</li>";
inline constexpr std::string_view open_list  = "<ul>";
// The only thing the server listens for.
inline constexpr std::string_view terminator = "</ul>";

struct result {
    bool                     complete       = false;
    size_t                   accepted_bytes = 0;
    std::vector<std::string> items;
    std::string              error;
};

// GBNF for the tokens generated after probe_prefix (not a whole document).
// The parser below stays the fail-closed authority.
std::string grammar();

// Returns string_view::npos while the terminator has not arrived, else the
// offset of its first byte. `begin` is the previous buffer length for an
// O(new bytes) incremental scan; the overlap needed for a terminator split
// across token pieces is applied internally.
size_t stop_offset(std::string_view text, size_t begin = 0);

// Parses the full wire, i.e. probe_prefix followed by generated bytes.
// Fail-closed: markup that is not exactly this envelope, an empty or
// whitespace-only question, or (with eof) a list that never closed, all become
// errors. Nothing is repaired, dropped or reordered. Leading/trailing whitespace
// of an item is trimmed; text after the terminator is discarded.
result parse(std::string_view text, bool eof = false);

// Canonical formal-P render: one question per line inside the same envelope.
std::string serialize(const std::vector<std::string> & items);

}  // namespace server_thinking
