# Selected-expert microfixture rejected before submission

The first six scalar shapes completed both modes and CPU checks. The next case
failed at `ggml_mul_mat` graph construction because the test supplied weight
batch10 and activation batch1; ggml only broadcasts in the opposite direction.
No tensor was submitted to GPU in that failed process; this is not a model or
driver failure. The old log and exit -6 remain preserved.

The standalone fixture now supplies one independent activation row per expert
batch and checks each batch against its matching CPU row. The three expert
microcases will be recorded in `service-experts/`, not substituted into old logs.
This is still generic batched matvec, not the model's fused MUL_MAT_ID region.
