# Framework Matrix Extended

- Generated: 2026-05-15 13:17:12 UTC
- Reference baseline document: [`framework-matrix-20260505-051038-corrected-with-lightautods.md`](/Users/a99/Downloads/5703-CS33-2/tests/system_reports/framework-matrix-20260505-051038-corrected-with-lightautods.md)
- This report is a fresh rerun with repeatability and token-usage instrumentation.
- Metric fields use 3-run aggregates per framework-task unless explicitly noted otherwise.

## Dataset Winners

| Dataset | Primary Metric | Better | Advantage |
| --- | --- | --- | ---: |
| `spaceship_titanic` | `accuracy` | `main` | 0.0039 |
| `academic_success` | `f1_macro` | `lightautods_lama` | 0.0002 |
| `house_prices` | `neg_root_mean_squared_error` | `aide` | 30644.9232 |
| `plate_defect` | `roc_auc_macro_ovr` | `lightautods_lama` | 0.0103 |
| `titanic_autokaggle` | `roc_auc` | `main` | 0.0075 |
| `bank_churn` | `roc_auc` | `autokaggle` | 0.0016 |
| `ghouls` | `f1_macro` | `lightautods_lama` | 0.0510 |
| `obesity_risks` | `f1_macro` | `autokaggle` | 0.0008 |

## Framework Metrics

| Dataset | Framework | Primary Metric | Score Mean | Reproducibility Std | Runtime Mean (s) | Runtime Std | Valid Output | Input Token Cost | Output Token Cost | Token Total | LLM Call Count |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `spaceship_titanic` | `main` | `accuracy` | 0.8003 | 0.0044 | 131.2004 | 37.4030 | `3/3` | 22202.6667 | 4997.6667 | 27200.3333 | 9.0000 |
| `spaceship_titanic` | `aide` | `accuracy` | 0.7762 | 0.0349 | 130.2890 | 59.1001 | `3/3` | 5486.3333 | 4460.6667 | 9947.0000 | 4.0000 |
| `spaceship_titanic` | `lightautods_lama` | `accuracy` | 0.7964 | 0.0000 | 14.0524 | 1.3013 | `3/3` | 816.0000 | 114.0000 | 930.0000 | 1.0000 |
| `spaceship_titanic` | `autokaggle` | `accuracy` | 0.7792 | 0.0000 | 38.2415 | 9.0722 | `3/3` | 1327.3333 | 135.6667 | 1463.0000 | 1.0000 |
| `academic_success` | `main` | `f1_macro` | 0.7927 | 0.0002 | 1065.2333 | 228.2234 | `3/3` | 84543.0000 | 4809.6667 | 89352.6667 | 9.0000 |
| `academic_success` | `aide` | `f1_macro` | 0.7878 | 0.0014 | 752.0823 | 420.5276 | `3/3` | 5871.0000 | 5076.3333 | 10947.3333 | 4.3333 |
| `academic_success` | `lightautods_lama` | `f1_macro` | 0.7929 | 0.0000 | 134.8703 | 25.1635 | `3/3` | 1544.0000 | 222.6667 | 1766.6667 | 1.0000 |
| `academic_success` | `autokaggle` | `f1_macro` | 0.7909 | 0.0000 | 451.2764 | 198.1943 | `3/3` | 1431.6667 | 189.3333 | 1621.0000 | 1.0000 |
| `house_prices` | `main` | `neg_root_mean_squared_error` | -30544.4876 | 278.6330 | 106.0116 | 15.0950 | `3/3` | 92185.3333 | 5370.3333 | 97555.6667 | 9.0000 |
| `house_prices` | `aide` | `neg_root_mean_squared_error` | 2676.9567 | 30924.1638 | 116.5753 | 55.2713 | `3/3` | 4268.0000 | 4091.3333 | 8359.3333 | 4.0000 |
| `house_prices` | `lightautods_lama` | `neg_root_mean_squared_error` | -30231.3257 | 0.0000 | 29.1273 | 8.0251 | `3/3` | 2356.3333 | 851.3333 | 3207.6667 | 1.0000 |
| `house_prices` | `autokaggle` | `neg_root_mean_squared_error` | -27967.9664 | 0.0000 | 42.3583 | 3.5149 | `3/3` | 1646.0000 | 261.0000 | 1907.0000 | 1.0000 |
| `plate_defect` | `main` | `roc_auc_macro_ovr` | 0.8752 | 0.0009 | 225.0026 | 18.7078 | `3/3` | 64717.6667 | 4732.0000 | 69449.6667 | 9.0000 |
| `plate_defect` | `aide` | `roc_auc_macro_ovr` | 0.8658 | 0.0101 | 192.8497 | 100.2145 | `3/3` | 6133.6667 | 6245.3333 | 12379.0000 | 4.0000 |
| `plate_defect` | `lightautods_lama` | `roc_auc_macro_ovr` | 0.8855 | 0.0000 | 198.5359 | 22.3677 | `3/3` | 1558.3333 | 554.6667 | 2113.0000 | 1.0000 |
| `plate_defect` | `autokaggle` | `roc_auc_macro_ovr` | 0.8748 | 0.0000 | 243.7002 | 17.8951 | `3/3` | 4832.3333 | 962.6667 | 5795.0000 | 3.0000 |
| `titanic_autokaggle` | `main` | `roc_auc` | 0.8758 | 0.0024 | 92.7997 | 20.0901 | `3/3` | 20088.0000 | 4563.0000 | 24651.0000 | 9.0000 |
| `titanic_autokaggle` | `aide` | `roc_auc` | 0.8684 | 0.0184 | 138.0435 | 59.2483 | `3/3` | 5726.3333 | 7286.0000 | 13012.3333 | 4.0000 |
| `titanic_autokaggle` | `lightautods_lama` | `roc_auc` | 0.7037 | 0.0000 | 10.5659 | 5.0415 | `3/3` | 736.0000 | 208.3333 | 944.3333 | 1.0000 |
| `titanic_autokaggle` | `autokaggle` | `roc_auc` | 0.8522 | 0.0000 | 25.8211 | 3.9880 | `3/3` | 1306.0000 | 241.6667 | 1547.6667 | 1.0000 |
| `bank_churn` | `main` | `roc_auc` | 0.8890 | 0.0001 | 327.5742 | 83.2482 | `3/3` | 21197.6667 | 4551.0000 | 25748.6667 | 9.0000 |
| `bank_churn` | `aide` | `roc_auc` | 0.8831 | 0.0083 | 125.2301 | 71.8508 | `3/3` | 4783.0000 | 3692.3333 | 8475.3333 | 4.0000 |
| `bank_churn` | `lightautods_lama` | `roc_auc` | 0.8771 | 0.0000 | 59.8585 | 8.4408 | `3/3` | 784.0000 | 323.3333 | 1107.3333 | 1.0000 |
| `bank_churn` | `autokaggle` | `roc_auc` | 0.8907 | 0.0000 | 363.0899 | 38.1489 | `3/3` | 1297.0000 | 224.6667 | 1521.6667 | 1.0000 |
| `ghouls` | `main` | `f1_macro` | 0.7566 | 0.0049 | 87.6796 | 18.6859 | `3/3` | 17176.3333 | 4587.0000 | 21763.3333 | 9.0000 |
| `ghouls` | `aide` | `f1_macro` | 0.7141 | 0.0196 | 82.6104 | 22.6843 | `3/3` | 4057.6667 | 3136.3333 | 7194.0000 | 4.0000 |
| `ghouls` | `lightautods_lama` | `f1_macro` | 0.8366 | 0.0000 | 16.0525 | 3.7147 | `3/3` | 622.0000 | 284.0000 | 906.0000 | 1.0000 |
| `ghouls` | `autokaggle` | `f1_macro` | 0.7857 | 0.0000 | 26.3554 | 2.7528 | `3/3` | 1249.0000 | 290.6667 | 1539.6667 | 1.0000 |
| `obesity_risks` | `main` | `f1_macro` | 0.8924 | 0.0007 | 199.4164 | 9.0046 | `3/3` | 23635.6667 | 5638.3333 | 29274.0000 | 9.0000 |
| `obesity_risks` | `aide` | `f1_macro` | 0.8883 | 0.0004 | 157.1564 | 41.7897 | `2/3` | 3223.0000 | 4941.6667 | 8164.6667 | 5.6667 |
| `obesity_risks` | `lightautods_lama` | `f1_macro` | 0.8881 | 0.0016 | 81.4080 | 10.2755 | `3/3` | 913.0000 | 255.3333 | 1168.3333 | 1.0000 |
| `obesity_risks` | `autokaggle` | `f1_macro` | 0.8932 | 0.0000 | 125.7852 | 15.0081 | `3/3` | 1313.0000 | 114.6667 | 1427.6667 | 1.0000 |

## `spaceship_titanic`

- Description: `single-target binary classification`
- Primary metric: `accuracy`
- Better framework: `main`
- Advantage: `0.0039`

### main

- Score mean: `0.8003`
- Reproducibility std: `0.0044`
- Runtime mean (s): `131.2004`
- Runtime std: `37.4030`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `22202.6667`
- Output token cost: `4997.6667`
- Token total: `27200.3333`
- LLM call count: `9.0000`
- Representative prediction file: `artifacts/runs/spaceship_titanic_20260515_173317_dbd1ff8b/predictions/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/spaceship_titanic_20260515_173317_dbd1ff8b`

### aide

- Score mean: `0.7762`
- Reproducibility std: `0.0349`
- Runtime mean (s): `130.2890`
- Runtime std: `59.1001`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `5486.3333`
- Output token cost: `4460.6667`
- Token total: `9947.0000`
- LLM call count: `4.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-popular-cheerful-boar/working/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/logs/2-popular-cheerful-boar`

### lightautods_lama

- Score mean: `0.7964`
- Reproducibility std: `0.0000`
- Runtime mean (s): `14.0524`
- Runtime std: `1.3013`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `816.0000`
- Output token cost: `114.0000`
- Token total: `930.0000`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-spaceship_titanic-20260515-093630-622038/models/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-spaceship_titanic-20260515-093630-622038`

### autokaggle

- Score mean: `0.7792`
- Reproducibility std: `0.0000`
- Runtime mean (s): `38.2415`
- Runtime std: `9.0722`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `1327.3333`
- Output token cost: `135.6667`
- Token total: `1463.0000`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-spaceship_titanic-20260515-093646-892585/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-spaceship_titanic-20260515-093646-892585`


## `academic_success`

- Description: `single-target multiclass classification`
- Primary metric: `f1_macro`
- Better framework: `lightautods_lama`
- Advantage: `0.0002`

### main

- Score mean: `0.7927`
- Reproducibility std: `0.0002`
- Runtime mean (s): `1065.2333`
- Runtime std: `228.2234`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `84543.0000`
- Output token cost: `4809.6667`
- Token total: `89352.6667`
- LLM call count: `9.0000`
- Representative prediction file: `artifacts/runs/classification_with_an_academic_success_dataset_20260515_173800_d18ac345/predictions/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/classification_with_an_academic_success_dataset_20260515_173800_d18ac345`

### aide

- Score mean: `0.7878`
- Reproducibility std: `0.0014`
- Runtime mean (s): `752.0823`
- Runtime std: `420.5276`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `5871.0000`
- Output token cost: `5076.3333`
- Token total: `10947.3333`
- LLM call count: `4.3333`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-funky-sarcastic-silkworm/working/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/logs/2-funky-sarcastic-silkworm`

### lightautods_lama

- Score mean: `0.7929`
- Reproducibility std: `0.0000`
- Runtime mean (s): `134.8703`
- Runtime std: `25.1635`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `1544.0000`
- Output token cost: `222.6667`
- Token total: `1766.6667`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-academic_success-20260515-094521-090587/models/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-academic_success-20260515-094521-090587`

### autokaggle

- Score mean: `0.7909`
- Reproducibility std: `0.0000`
- Runtime mean (s): `451.2764`
- Runtime std: `198.1943`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `1431.6667`
- Output token cost: `189.3333`
- Token total: `1621.0000`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-academic_success-20260515-094738-585441/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-academic_success-20260515-094738-585441`


## `house_prices`

- Description: `single-target regression`
- Primary metric: `neg_root_mean_squared_error`
- Better framework: `aide`
- Advantage: `30644.9232`

### main

- Score mean: `-30544.4876`
- Reproducibility std: `278.6330`
- Runtime mean (s): `106.0116`
- Runtime std: `15.0950`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `92185.3333`
- Output token cost: `5370.3333`
- Token total: `97555.6667`
- LLM call count: `9.0000`
- Representative prediction file: `artifacts/runs/house_prices_20260515_175848_002d3ca2/predictions/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/house_prices_20260515_175848_002d3ca2`

### aide

- Score mean: `2676.9567`
- Reproducibility std: `30924.1638`
- Runtime mean (s): `116.5753`
- Runtime std: `55.2713`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `4268.0000`
- Output token cost: `4091.3333`
- Token total: `8359.3333`
- LLM call count: `4.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-psychedelic-roaring-dragon/working/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/logs/2-psychedelic-roaring-dragon`

### lightautods_lama

- Score mean: `-30231.3257`
- Reproducibility std: `0.0000`
- Runtime mean (s): `29.1273`
- Runtime std: `8.0251`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `2356.3333`
- Output token cost: `851.3333`
- Token total: `3207.6667`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-house_prices-20260515-100244-873604/models/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-house_prices-20260515-100244-873604`

### autokaggle

- Score mean: `-27967.9664`
- Reproducibility std: `0.0000`
- Runtime mean (s): `42.3583`
- Runtime std: `3.5149`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `1646.0000`
- Output token cost: `261.0000`
- Token total: `1907.0000`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-house_prices-20260515-100306-543913/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-house_prices-20260515-100306-543913`


## `plate_defect`

- Description: `multi-target multilabel classification`
- Primary metric: `roc_auc_macro_ovr`
- Better framework: `lightautods_lama`
- Advantage: `0.0103`

### main

- Score mean: `0.8752`
- Reproducibility std: `0.0009`
- Runtime mean (s): `225.0026`
- Runtime std: `18.7078`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `64717.6667`
- Output token cost: `4732.0000`
- Token total: `69449.6667`
- LLM call count: `9.0000`
- Representative prediction file: `artifacts/runs/plate_defect_20260515_180350_857ac151/predictions/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/plate_defect_20260515_180350_857ac151`

### aide

- Score mean: `0.8658`
- Reproducibility std: `0.0101`
- Runtime mean (s): `192.8497`
- Runtime std: `100.2145`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `6133.6667`
- Output token cost: `6245.3333`
- Token total: `12379.0000`
- LLM call count: `4.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-healthy-wine-petrel/working/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/logs/2-healthy-wine-petrel`

### lightautods_lama

- Score mean: `0.8855`
- Reproducibility std: `0.0000`
- Runtime mean (s): `198.5359`
- Runtime std: `22.3677`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `1558.3333`
- Output token cost: `554.6667`
- Token total: `2113.0000`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-plate_defect-20260515-100856-425471/models/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-plate_defect-20260515-100856-425471`

### autokaggle

- Score mean: `0.8748`
- Reproducibility std: `0.0000`
- Runtime mean (s): `243.7002`
- Runtime std: `17.8951`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `4832.3333`
- Output token cost: `962.6667`
- Token total: `5795.0000`
- LLM call count: `3.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-plate_defect-20260515-101054-732184/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-plate_defect-20260515-101054-732184`


## `titanic_autokaggle`

- Description: `single-target binary classification smoke dataset`
- Primary metric: `roc_auc`
- Better framework: `main`
- Advantage: `0.0075`

### main

- Score mean: `0.8758`
- Reproducibility std: `0.0024`
- Runtime mean (s): `92.7997`
- Runtime std: `20.0901`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `20088.0000`
- Output token cost: `4563.0000`
- Token total: `24651.0000`
- LLM call count: `9.0000`
- Representative prediction file: `artifacts/runs/titanic_autokaggle_20260515_181201_67d3e371/predictions/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/titanic_autokaggle_20260515_181201_67d3e371`

### aide

- Score mean: `0.8684`
- Reproducibility std: `0.0184`
- Runtime mean (s): `138.0435`
- Runtime std: `59.2483`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `5726.3333`
- Output token cost: `7286.0000`
- Token total: `13012.3333`
- LLM call count: `4.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-vague-shrimp-of-promise/working/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/logs/2-vague-shrimp-of-promise`

### lightautods_lama

- Score mean: `0.7037`
- Reproducibility std: `0.0000`
- Runtime mean (s): `10.5659`
- Runtime std: `5.0415`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `736.0000`
- Output token cost: `208.3333`
- Token total: `944.3333`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-titanic_autokaggle-20260515-101533-303723/models/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-titanic_autokaggle-20260515-101533-303723`

### autokaggle

- Score mean: `0.8522`
- Reproducibility std: `0.0000`
- Runtime mean (s): `25.8211`
- Runtime std: `3.9880`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `1306.0000`
- Output token cost: `241.6667`
- Token total: `1547.6667`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-titanic_autokaggle-20260515-101550-163245/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-titanic_autokaggle-20260515-101550-163245`


## `bank_churn`

- Description: `single-target binary classification smoke dataset`
- Primary metric: `roc_auc`
- Better framework: `autokaggle`
- Advantage: `0.0016`

### main

- Score mean: `0.8890`
- Reproducibility std: `0.0001`
- Runtime mean (s): `327.5742`
- Runtime std: `83.2482`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `21197.6667`
- Output token cost: `4551.0000`
- Token total: `25748.6667`
- LLM call count: `9.0000`
- Representative prediction file: `artifacts/runs/bank_churn_20260515_181622_7977ed91/predictions/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/bank_churn_20260515_181622_7977ed91`

### aide

- Score mean: `0.8831`
- Reproducibility std: `0.0083`
- Runtime mean (s): `125.2301`
- Runtime std: `71.8508`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `4783.0000`
- Output token cost: `3692.3333`
- Token total: `8475.3333`
- LLM call count: `4.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-quick-avocet-from-valhalla/working/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/logs/2-quick-avocet-from-valhalla`

### lightautods_lama

- Score mean: `0.8771`
- Reproducibility std: `0.0000`
- Runtime mean (s): `59.8585`
- Runtime std: `8.4408`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `784.0000`
- Output token cost: `323.3333`
- Token total: `1107.3333`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-bank_churn-20260515-101811-033668/models/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-bank_churn-20260515-101811-033668`

### autokaggle

- Score mean: `0.8907`
- Reproducibility std: `0.0000`
- Runtime mean (s): `363.0899`
- Runtime std: `38.1489`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `1297.0000`
- Output token cost: `224.6667`
- Token total: `1521.6667`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-bank_churn-20260515-101908-749321/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-bank_churn-20260515-101908-749321`


## `ghouls`

- Description: `single-target multiclass classification smoke dataset`
- Primary metric: `f1_macro`
- Better framework: `lightautods_lama`
- Advantage: `0.0510`

### main

- Score mean: `0.7566`
- Reproducibility std: `0.0049`
- Runtime mean (s): `87.6796`
- Runtime std: `18.6859`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `17176.3333`
- Output token cost: `4587.0000`
- Token total: `21763.3333`
- LLM call count: `9.0000`
- Representative prediction file: `artifacts/runs/ghouls_goblins_and_ghosts_boo_20260515_182451_5c5281fa/predictions/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/ghouls_goblins_and_ghosts_boo_20260515_182451_5c5281fa`

### aide

- Score mean: `0.7141`
- Reproducibility std: `0.0196`
- Runtime mean (s): `82.6104`
- Runtime std: `22.6843`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `4057.6667`
- Output token cost: `3136.3333`
- Token total: `7194.0000`
- LLM call count: `4.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-savvy-cherry-giraffe/working/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/logs/2-savvy-cherry-giraffe`

### lightautods_lama

- Score mean: `0.8366`
- Reproducibility std: `0.0000`
- Runtime mean (s): `16.0525`
- Runtime std: `3.7147`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `622.0000`
- Output token cost: `284.0000`
- Token total: `906.0000`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-ghouls-20260515-102718-030871/models/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-ghouls-20260515-102718-030871`

### autokaggle

- Score mean: `0.7857`
- Reproducibility std: `0.0000`
- Runtime mean (s): `26.3554`
- Runtime std: `2.7528`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `1249.0000`
- Output token cost: `290.6667`
- Token total: `1539.6667`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-ghouls-20260515-102727-179190/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-ghouls-20260515-102727-179190`


## `obesity_risks`

- Description: `single-target multiclass classification smoke dataset`
- Primary metric: `f1_macro`
- Better framework: `autokaggle`
- Advantage: `0.0008`

### main

- Score mean: `0.8924`
- Reproducibility std: `0.0007`
- Runtime mean (s): `199.4164`
- Runtime std: `9.0046`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `23635.6667`
- Output token cost: `5638.3333`
- Token total: `29274.0000`
- LLM call count: `9.0000`
- Representative prediction file: `artifacts/runs/obesity_risks_20260515_182740_e651485b/predictions/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/artifacts/runs/obesity_risks_20260515_182740_e651485b`

### aide

- Score mean: `0.8883`
- Reproducibility std: `0.0004`
- Runtime mean (s): `157.1564`
- Runtime std: `41.7897`
- Valid output rate: `2/3`
- Success rate: `2/3`
- Input token cost: `3223.0000`
- Output token cost: `4941.6667`
- Token total: `8164.6667`
- LLM call count: `5.6667`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/workspaces/2-discreet-likable-koala/working/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/aide_eval/aideml/logs/2-discreet-likable-koala`

### lightautods_lama

- Score mean: `0.8881`
- Reproducibility std: `0.0016`
- Runtime mean (s): `81.4080`
- Runtime std: `10.2755`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `913.0000`
- Output token cost: `255.3333`
- Token total: `1168.3333`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-obesity_risks-20260515-103101-230664/models/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/tests/system_reports/tmp/lads-lama-obesity_risks-20260515-103101-230664`

### autokaggle

- Score mean: `0.8932`
- Reproducibility std: `0.0000`
- Runtime mean (s): `125.7852`
- Runtime std: `15.0081`
- Valid output rate: `3/3`
- Success rate: `3/3`
- Input token cost: `1313.0000`
- Output token cost: `114.6667`
- Token total: `1427.6667`
- LLM call count: `1.0000`
- Representative prediction file: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-obesity_risks-20260515-103202-895502/submission.csv`
- Representative run dir: `/Users/a99/Downloads/5703-CS33-2/_external/autokaggle_eval/AutoKaggle/multi_agents/competition/autokaggle-obesity_risks-20260515-103202-895502`

## Rendering Notes

- Raw run rows found: `95`
- Recovered rows appended from existing logs: `1`
- Recovered `aide` / `obesity_risks` / `r3` as `FAIL` from existing usage logs only.
