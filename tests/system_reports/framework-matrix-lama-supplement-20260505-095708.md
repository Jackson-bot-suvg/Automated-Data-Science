# LightAutoDS LAMA Supplement

- Generated: `2026-05-05 09:57:08 UTC`
- Baseline source: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/framework-matrix-20260505-051038-corrected.json`
- Comparison scope: existing `main` + `AIDE` results were reused; only `LightAutoDS` was rerun.

## Snapshot

- Compared datasets: `2`
- Winner count: `main=1`, `aide=0`, `lightautods_lama=1`
- Total runtime (s): `main=288.4580`, `aide=217.3718`, `lightautods_lama=11.3452`
- LightAutoDS route note: `LightAutoML-only`, scored by our external metric on held-out validation predictions; not the full CodeGen/FEDOT graph.
- Stability note: this run uses a single-process LightAutoML preset for compatibility on the current machine.

## Scoreboard

| Dataset | Metric | Main | AIDE | LightAutoDS LAMA | Best | Advantage | Main Time (s) | AIDE Time (s) | LAMA Time (s) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `titanic_autokaggle` | `roc_auc` | 0.8745 | 0.8660 | 0.7037 | `main` | 0.0085 | 110.9870 | 114.8955 | 3.1539 |
| `house_prices` | `neg_root_mean_squared_error` | -31606.2974 | -32195.9950 | -30231.3257 | `lightautods_lama` | 1374.9717 | 177.4710 | 102.4763 | 8.1913 |

## 1. `titanic_autokaggle`

- Description: single-target binary classification smoke dataset
- Main metric: `roc_auc`
- Best framework: `main`
- Score delta vs best: `0.0085`
- Main score/runtime: `0.8745` / `110.9870s`
- AIDE score/runtime: `0.8660` / `114.8955s`
- LightAutoDS LAMA score/runtime: `0.7037` / `3.1539s`
- LightAutoDS route: `LAMA`
- LightAutoDS internal config: `{"task_type": "binary", "target": "Survived", "task_metric": "auc"}`
- LightAutoDS validation predictions: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-titanic_autokaggle-20260505-095656/models/validation_predictions.csv`
- LightAutoDS submission: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-titanic_autokaggle-20260505-095656/models/submission.csv`
- Main prediction file: `artifacts/runs/titanic_autokaggle_20260505_154202_d35ee0ab/predictions/submission.csv`
- AIDE prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-phenomenal-quick-ladybug/working/submission.csv`

## 2. `house_prices`

- Description: single-target regression
- Main metric: `neg_root_mean_squared_error`
- Best framework: `lightautods_lama`
- Score delta vs best: `1374.9717`
- Main score/runtime: `-31606.2974` / `177.4710s`
- AIDE score/runtime: `-32195.9950` / `102.4763s`
- LightAutoDS LAMA score/runtime: `-30231.3257` / `8.1913s`
- LightAutoDS route: `LAMA`
- LightAutoDS internal config: `{"task_type": "reg", "target": "SalePrice", "task_metric": "r2"}`
- LightAutoDS validation predictions: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-house_prices-20260505-095659/models/validation_predictions.csv`
- LightAutoDS submission: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-house_prices-20260505-095659/models/submission.csv`
- Main prediction file: `artifacts/runs/house_prices_20260505_152833_82bd7d09/predictions/submission.csv`
- AIDE prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-rich-divergent-puffin/working/submission.csv`
