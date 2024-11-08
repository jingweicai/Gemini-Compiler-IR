# ZeBu Trace Output Format

This trace captures the start and end times of each instruction executed on ZEBU. We use these results to validate against our simulator.

The ZeBu trace format is similar to that of the assembler, with the addition of an 'inst_idx' and execution 'time': [start, end] for each instruction. Note that instructions can be dispatched out-of-order and multiple at a time, so the execution times may overlap.


**Example**:

```json
[lda_cfg, {...}, {'inst_idx': 6, 'time': [0, 21]}, {...}]
[lda_mov, {...}, {'inst_idx': 7, 'time': [175, 23563]}, {...}]
```
`lda_cfg` has an `'inst_idx'` of 6 and starts execution at time 0 and finishes at time 21. It occupies 21 units of time. 
`lda_mov` has an `'inst_idx'` of 7 and starts execution at time 175 and finishes at time 23563. It takes a much longer time, indicating a long-running data movement operation.
