# Bench results

- created: 2026-08-16T00:29:54+00:00
- store: `work`
- args: n_features=20 n_tokens=20 trials=5 seed=0
- cache drop method: **drop_caches**
- cpu: Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz x30, ram: 222.2 GiB
- versions: python 3.10.12, duckdb 1.5.5, pyarrow 25.0.1

```
NAME   ROTA MODEL
loop0     1 
loop1     1 
loop2     1 
loop3     1 
loop4     1 
loop5     1 
loop6     1 
loop7     1 
loop8     1 
loop9     1 
loop10    1 
vda       1 
vdb       1
```

## tokens_for_feature (cold)

| method | p50 (s) | p90 (s) | p99 (s) | mean (s) | rows | bytes read | files | row groups |
|---|---|---|---|---|---|---|---|---|
| bucketed | 0.0532 | 0.0633 | 0.0807 | 0.0538 | 134,704 | 7.1 MiB | 1.0 | 1.1 |
| flat_pyarrow | 1.4643 | 1.4881 | 1.8632 | 1.4756 | 134,704 | 5.5 GiB | 20.0 | 1046.0 |
| flat_duckdb | 1.1239 | 1.2767 | 1.4078 | 1.1266 | 134,704 | 5.5 GiB | 20.0 | - |
| token_bucketed_scan | 1.5539 | 1.5727 | 1.8511 | 1.5629 | 134,704 | 4.8 GiB | 128.0 | 1152.0 |

## tokens_for_feature (warm)

| method | p50 (s) | p90 (s) | p99 (s) | mean (s) | rows | bytes read | files | row groups |
|---|---|---|---|---|---|---|---|---|
| bucketed | 0.0149 | 0.0166 | 0.0200 | 0.0142 | 134,704 | 7.1 MiB | 1.0 | 1.1 |
| flat_pyarrow | 1.0155 | 1.0354 | 1.0523 | 1.0168 | 134,704 | 5.5 GiB | 20.0 | 1046.0 |
| flat_duckdb | 0.5349 | 0.6978 | 0.8425 | 0.5335 | 134,704 | 5.5 GiB | 20.0 | - |
| token_bucketed_scan | 1.1949 | 1.2105 | 1.2228 | 1.1954 | 134,704 | 4.8 GiB | 128.0 | 1152.0 |

## features_for_token (cold)

| method | p50 (s) | p90 (s) | p99 (s) | mean (s) | rows | bytes read | files | row groups |
|---|---|---|---|---|---|---|---|---|
| feature_bucketed_scan | 1.5659 | 1.6017 | 1.6158 | 1.5686 | 32 | 5.7 GiB | 128.0 | 968.6 |
| flat_pyarrow | 0.0390 | 0.0691 | 0.0737 | 0.0415 | 32 | 5.0 MiB | 1.0 | 1.0 |
| flat_duckdb | 0.1532 | 0.1565 | 0.1624 | 0.1531 | 32 | 5.5 GiB | 20.0 | - |
| token_bucketed | 0.0255 | 0.0491 | 0.0514 | 0.0295 | 32 | 4.5 MiB | 1.0 | 1.0 |

## features_for_token (warm)

| method | p50 (s) | p90 (s) | p99 (s) | mean (s) | rows | bytes read | files | row groups |
|---|---|---|---|---|---|---|---|---|
| feature_bucketed_scan | 1.1437 | 1.1619 | 1.1740 | 1.1418 | 32 | 5.7 GiB | 128.0 | 968.6 |
| flat_pyarrow | 0.0167 | 0.0229 | 0.0249 | 0.0173 | 32 | 5.0 MiB | 1.0 | 1.0 |
| flat_duckdb | 0.1198 | 0.1215 | 0.1252 | 0.1200 | 32 | 5.5 GiB | 20.0 | - |
| token_bucketed | 0.0108 | 0.0128 | 0.0138 | 0.0109 | 32 | 4.5 MiB | 1.0 | 1.0 |

## Kill point

```
KILL-POINT tokens_for_feature cold p99: bucketed 0.0807s vs flat scan 1.8632s -> 23.1x (threshold >= 5.0x): PASS
```
