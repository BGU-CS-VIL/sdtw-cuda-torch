# Barycenter analysis: softdba vs classic DBA (NVIDIA GeForce RTX 3090)

torch 2.6.0+cu124, CUDA 12.4, gamma=1.0, softdba 100 iters, DBA 30 iters. `mean_dtw_cost` is the shared yardstick (lower = tighter alignment).

### K=32, L=128, D=1
| Method | Wall (s) | Peak mem (MB) | Mean DTW cost | Mean soft-DTW |
|---|--:|--:|--:|--:|
| softdba_fused | 3.94 | 24 | 8.441 | -186.165 |
| softdba_unfused | 0.26 | 31 | 8.441 | -186.165 |
| dba_exact | 0.00 | 24 | 9.659 | -141.777 |

### K=32, L=512, D=1
| Method | Wall (s) | Peak mem (MB) | Mean DTW cost | Mean soft-DTW |
|---|--:|--:|--:|--:|
| softdba_fused | 15.01 | 145 | 24.535 | -787.052 |
| softdba_unfused | 1.91 | 241 | 24.535 | -787.052 |
| dba_exact | 0.03 | 144 | 31.908 | -575.630 |
