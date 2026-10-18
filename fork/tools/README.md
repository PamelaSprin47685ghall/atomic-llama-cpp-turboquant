# Independent comparison tools

These tools use only the Python standard library. They do not load a model,
initialize a GPU, or depend on private llama.cpp headers.

## Full-logits comparison

Run with Python 3.9 or newer:

    python3 fork/tools/compare-logits.py reference.bin candidate.bin \
        --atol 0.0001 --rtol 0.00001 --output comparison.json

The reader retains the original TP5LOG2 file magic for compatibility with
existing fixed-input tapes. It compares every vocabulary logit, one row at a
time, and requires identical declared dimensions and teacher-forced input
token IDs. Memory grows with one vocabulary row and the input token list,
not with all saved logits.

Exit code 0 means that every compared value is finite and satisfies

    abs(candidate - reference) <= atol + rtol * abs(reference)

Exit code 1 means a numerical comparison, format, input, or output failure;
invalid command-line arguments use exit code 2. Existing report paths are
refused, including the input tape paths. Errors are reported without a
traceback. The Python compare() API applies the same tolerance validation.

The JSON reports bitwise equality, differing finite values, out-of-tolerance
values, maximum absolute and relative errors, normalized squared error, and
argmax disagreements. Numerical tolerance is the pass criterion; bitwise
and argmax agreement are separate metrics. For example, signed zeros may
compare numerically equal with different bytes, and a permitted logit error
can change an argmax. A row containing a nonfinite value always fails and
is counted in argmax_unavailable_rows rather than reported as a valid
argmax comparison. The nonfinite count counts coordinates where either
input is nonfinite, not the number of individual nonfinite values.

max_rel uses max(abs(reference), 1e-12) as its denominator.
normalized_mse is sum((candidate-reference)^2)/sum(reference^2);
an all-zero reference with nonzero error yields JSON null, and two
all-zero tapes yield zero. For invalid nonfinite rows, error statistics
cover only the finite coordinate pairs and do not establish agreement.

### Tape format

All fields are little endian:

- 8 bytes: TP5LOG2 followed by a zero byte.
- Four uint32 values: vocabulary size, prefill token count, decode step
  count, and total input token count.
- total_input_count int32 token IDs.
- (decode_steps + 1) rows of vocabulary_size float32 logits.

total_input_count must equal prefill_count + decode_steps. Prefill is
nonempty; zero decode steps represent a valid prefill-only tape. Vocabulary
and input counts are each limited to 2,000,000. Every input token must be
inside the declared vocabulary. Short sections and trailing data are
rejected. The reader requires a host Python with 32-bit float arrays and
handles big-endian hosts explicitly.

The tape does not carry model, tokenizer, build, hardware, or context-policy
identity. Record that provenance beside the tapes when selecting a valid
comparison; identical token IDs and vocabulary sizes alone do not prove
tokenizer identity. This tool provides numerical evidence, not a benchmark
or a model quality score.

## Tests

    python3 fork/tools/tests/test-compare-logits.py

The tests invoke the real CLI and create real binary files in temporary
directories. They cover valid full-vocabulary comparisons, finite values,
thresholds, token and dimension identity, truncation, trailing data, and
output preservation. They do not mock the tape reader.
