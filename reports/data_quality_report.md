# Data Quality Report

## Audit scope

The source files are `data/raw/clientes_restaurantes.csv` and `data/raw/datas_comemorativas.csv`. The supplied demand data contains 478 daily rows from 2016-01-01 through 2017-04-22. The date sequence has no gaps or duplicates.

## Demand targets

| Target | Missing values | Observed minimum | Observed maximum | Observed mean |
|---|---:|---:|---:|---:|
| Chimi & Churri | 2 | 1 | 128 | 37.50 |
| Assa Frão | 1 | 3 | 139 | 37.61 |

Missing target values are unknown. No values were filled during this audit. Missingness flags are retained by the loading function.

## Commemorative calendar

The calendar has 17 unique dates, from 2016-01-01 through 2017-04-17. It covers only part of the demand span and does not cover the historical forecast horizon after 2017-04-22. Labels are retained as supplied. Their business meaning, local applicability, and production availability have not been verified; holiday effects are therefore not assumed.

## Reference forecast

The supplied workbook contains 28 forecast rows dated 2017-04-23 through 2017-05-20, immediately following the demand cutoff. Its generating method and evaluation provenance are unavailable, so its values are preserved as a reference artifact and are not treated as model results.

## Environment

The active Python is 3.12.3. The initial virtual environment was empty; after package-index access was enabled, the core dependencies installed successfully. All 12 numbered notebooks executed, and Prophet fitted and serialized with the installed CmdStan backend.

## Dependency installation attempt

The first sandboxed installation attempt could not resolve pypi.org (NameResolutionError: HTTPSConnection(host='pypi.org', port=443): Failed to resolve 'pypi.org' ([Errno -2] Name or service not known), followed by ERROR: No matching distribution found for pandas<3,>=2.2). Retrying with package-index access succeeded. Installed versions included pandas 2.3.3, NumPy 2.5.4, scikit-learn 1.9.1, Statsmodels 0.15.0, Prophet 1.5.0, and pytest 8.4.2.

