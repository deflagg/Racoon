# Information-efficiency confirmatory sweep

Date: 2026-10-14

## Setup

This repeats the targeted-vs-random information-efficiency curve with a larger seed count and a much larger fixed training budget.

- Frozen model: Raccoon, 244 trainable parameters
- Seeds: 12 (0 through 11)
- Optimizer: AdamW
- Learning rate: 3e-3
- Batch size: 24
- Training budget: 12,000 optimizer updates per condition
- Same paired seed initialization across conditions
- Same random-data seed as the pilot: 20261013
- Held-out conflict starts: 5, 6, 7
- Targeted counterfactual order: (mode0,start2), (mode1,start2), (mode0,start3), (mode1,start3), (mode0,start4), (mode1,start4)
- Random-novel examples use the same sequence ordering rule as the pilot.

The run used the parallel-seed baseline adopted for this study: parameters and AdamW state are independent per model slice; the parallelization is only computational batching.

## Final held-out conflict behavior at 12,000 updates

| Extra examples | Targeted: follow mode | Random novel: follow mode | Targeted advantage |
|---:|---:|---:|---:|
| 1 | 53.9% | 44.4% | +9.5 pp |
| 2 | 64.8% | 56.0% | +8.8 pp |
| 3 | 70.6% | 54.6% | +16.0 pp |
| 4 | 70.8% | 50.5% | +20.4 pp |
| 5 | 77.1% | 59.7% | +17.4 pp |
| 6 | 84.3% | 62.5% | +21.8 pp |

Baseline (no added examples): 40.7% mode-following and 47.7% trajectory-following.

At +6 targeted examples, trajectory-shortcut following fell to 9.7%, versus 26.2% for +6 random-novel examples.

## Paired seed bootstrap uncertainty

The comparison is paired by initialization seed. Bootstrap 95% CIs for the targeted-minus-random difference:

| k | Mean difference | 95% bootstrap CI | Seeds targeted > random |
|---:|---:|---:|---:|
| 1 | +9.5 pp | [-0.7, +20.1] | 7/12 |
| 2 | +8.8 pp | [-2.8, +21.3] | 7/12 |
| 3 | +16.0 pp | [+0.2, +30.1] | 9/12 |
| 4 | +20.4 pp | [+5.1, +36.6] | 8/12 |
| 5 | +17.4 pp | [+5.1, +30.3] | 10/12 |
| 6 | +21.8 pp | [+7.6, +35.4] | 10/12 |

## Optimization / convergence diagnostic

This was intentionally a larger fixed-budget run, not a strict all-seeds-to-zero-loss run. Mean training accuracy at 12,000 updates was high (roughly 96% to 100% across conditions), but not every seed reached perfect training accuracy in every condition. For example, +6 targeted had 9/12 perfect-fit seeds; +6 random had 4/12.

The targeted-vs-random separation is already stable from roughly 9,000 to 12,000 steps, but optimization remains a residual confound in the hardest conditions. Therefore the result supports an information-efficiency effect, but should not yet be treated as a pure identifiability theorem.

## Main result

The pilot effect survives a larger run. At equal added-sample count, deliberately discriminating/counterfactual examples consistently bias Raccoon toward the intended mode rule more efficiently than the particular random-novel examples used here. The effect becomes clearest from k=3 onward and is strongest at k=4..6.

This supports the working hypothesis that data quality here is not just sample count: examples differ in how strongly they eliminate competing rules.
