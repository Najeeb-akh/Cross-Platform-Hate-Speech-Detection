
## Table 1: Cross-dataset performance metrics for all nine train→test combinations

F1-Drop% measures relative degradation from in-domain performance (positive = worse, negative = better).

| Train → Test | Accuracy | Precision | Recall | F1 | ROC-AUC | F1-Drop% |
|---|---|---|---|---|---|---|
| Reddit → Reddit | 0.864 | 0.630 | 0.775 | 0.695 | 0.922 | 0.0 |
| Reddit → 4chan | 0.897 | 0.706 | 0.845 | 0.767 | 0.919 | -10.4 |
| Reddit → Twitter | 0.902 | 0.685 | 0.950 | 0.795 | 0.954 | -14.5 |
| 4chan → Reddit | 0.864 | 0.639 | 0.740 | 0.685 | 0.917 | +12.3 |
| 4chan → 4chan | 0.906 | 0.740 | 0.830 | 0.780 | 0.926 | 0.0 |
| 4chan → Twitter | 0.914 | 0.716 | 0.950 | 0.816 | 0.974 | -4.6 |
| Twitter → Reddit | 0.863 | 0.641 | 0.725 | 0.678 | 0.865 | +19.6 |
| Twitter → 4chan | 0.904 | 0.739 | 0.815 | 0.773 | 0.917 | +8.4 |
| Twitter → Twitter | 0.930 | 0.768 | 0.940 | 0.844 | 0.974 | 0.0 |


### Summary Table (F1 Scores and F1-Drop%):

| Train → Test | Reddit | 4chan | Twitter |
|---|---|---|---|
| Reddit →  | **0.695** (0.0%) | 0.767 (-10.4%) | 0.795 (-14.5%) |
| 4chan →  | 0.685 (+12.3%) | **0.780** (0.0%) | 0.816 (-4.6%) |
| Twitter →  | 0.678 (+19.6%) | 0.773 (+8.4%) | **0.844** (0.0%) |
