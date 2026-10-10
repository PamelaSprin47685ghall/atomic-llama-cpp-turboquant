The first native probe enumerated counters, then refused before device/pipeline
creation because the old shorthand labels did not match current names. Actual
names are `VRAM read size`, `VMEM Load Instructions`, `Waves`, `VALU Instructions`,
`SALU Instructions`, `GPU active cycles`; all reported storage float64/scope command.
The probe now selects these verified names and reports units too. Old refusal is
preserved. No counter values, kernel times or success were fabricated. Actual
query results are stored under `traffic-v2/`, with empty controls first.
