# total-emission-footprint

Case studies of the "total emission footprint" of various activities/policies,
built on FaIR (the Finite Amplitude Impulse Response climate model).

## Environment

Activate the conda/mamba env before running anything in this repo:

```
mamba activate total-emission-footprint
```

Defined in [environment.yml](environment.yml): python 3.10, fair=2.2.4,
numpy, pandas, xarray, matplotlib, netcdf4.

`.env` (loaded via `python-dotenv` in [fair_tools.py](fair_tools.py)) sets:
- `FAIR_CALIBRATION_DIR` — path to an external `fair-calibrate` checkout
  (the Smith et al. 2024 AR6 calibration archive: posteriors, harmonized
  SSP emissions, prior-run diagnostics). Not part of this repo; must exist
  on disk at that path.
- `POSTERIOR_SAMPLES=841` — size of the constrained posterior ensemble used
  for every FaIR run in this repo.
- `DATADIR` / `FIGDIR` — input/output data dirs (most scripts also use their
  own local `figures/` / `output/` paths directly).

## Repo structure

- [fair_tools.py](fair_tools.py) — shared FaIR engine, used by every case
  study. See below.
- [shipping.py](shipping.py) — IMO 2020 shipping sulfur-cap case study
  (documented in detail below).
- [car_sauna.py](car_sauna.py) / [sauna_scenarios.py](sauna_scenarios.py) —
  a second case study (car driving + sauna heating activity pulses). Not
  documented in depth here yet.
- `figures/` — PNG outputs. `output/` — CSV outputs (e.g. GTP tables).

## fair_tools.py reference

- `createConstrainedRuns(scenarios, year_end, forcings)` — builds a FaIR
  instance over the full 841-member AR6-constrained posterior ensemble
  (calibration v1.6.0), configured but not run. Loads from
  `FAIR_CALIBRATION_DIR`:
  - harmonized emissions `output/emissions/ssps_harmonized_1750-2499.nc`
    (CMIP7 historical 1750-2023, SSP-extended beyond)
  - solar/volcanic forcing timeseries, scaled per-config by the posterior's
    `forcing_scale[Solar]` / `forcing_scale[Volcanic]`
  - posterior parameters `output/posteriors/calibrated_constrained_parameters.csv`
    and species configs `output/posteriors/species_configs_properties.csv`
  - `forcings={'non-ghg': bool, 'non-co2-ghgs': bool}` lets you switch
    groups of species to their baseline (pre-industrial/natural) emissions
    instead of the scenario's, isolating CO2-only or GHG-only responses.
  - Calls `_clip_emissions_to_baseline` internally: floors non-CO2 GHG
    emissions at their per-config baseline (natural) rate. This exists
    because the SSP-harmonized file has a known artifact — CFC-115 goes
    negative from 2024 onward (RCMIP v5.1.0 vs. CMIP7 historical mismatch
    reconciled by `aneris`) — and negative anthropogenic emissions are
    unphysical for `step_concentration`.
- `rebase_temperature(f)` — shifts `f.temperature` to a 1850-1900 baseline
  using `calculate_timemean` (trapezoidal-weighted mean over timebounds, so
  the two half-weighted endpoints don't double-count).
- `compute_tcre()` — returns per-config TCRE (K/GtCO2) and a `sat_1000gtc`
  diagnostic (K at 1000 GtC cumulative emissions), both as 841-member
  `xarray.DataArray`s over `config`. Looked up from the calibration
  archive's *already-run* 1pctCO2 experiment (`TCR / 3670 GtCO2`,
  3670 GtCO2 = 1000 GtC at CO2 doubling — the same approximation IPCC AR6
  uses) rather than re-running 1pctCO2 here.
- `update_scenario_names(f, scenario_map)` — renames the `scenario`
  coordinate across every DataArray on a FAIR instance.

## Shipping case study (shipping.py)

**Question**: the IMO 2020 marine fuel sulfur cap cut ship SO2 emissions,
which had been masking some warming via aerosol cooling. What's the net
warming effect of removing that cooling, and what's its CO2-equivalent
emissions footprint at GTP (Global Temperature change Potential) horizons of
20/50/100 years?

**Setup**: baseline SSP2-4.5 (`ssp='ssp245'`), run 1750-2150
(`year_end=2150`) over all 841 posterior configs. The forcing perturbation
is injected by adding to the model's `'Volcanic'` forcing slot (reused
purely as a generic ERF carrier — not actual volcanic forcing, just an
available input channel that doesn't otherwise interact with the run) —
three variants:
- `f_imo_continuous` — constant `+71 mW/m²` from 2020 through 2301
- `f_imo_1yr` — same `71 mW/m²`, but only 2020-2021 (a single-year pulse,
  for comparing pulse vs. sustained forcing)
- `f_imo_gettelman` — `+0.12 W/m²`, the alternative literature value from
  Gettelman et al. (2024), for comparison against the 71 mW/m² figure

Each run is diffed against the baseline (`dsat_* = sat_* - sat_base`) to
get the IMO-attributable temperature response.

**GTP / CO2-equivalent methodology** — two methods are computed side by
side, deliberately, to quantify the error of the simpler one:
- **exact**: divide *each ensemble member's own* ΔT by *that same member's*
  own TCRE, then average across members. Preserves the real per-config
  correlation between climate sensitivity and TCRE (a high-TCRE config also
  tends to warm faster, so this correlation matters).
- **shortcut**: divide the ensemble-*mean* ΔT by the ensemble-*mean* TCRE —
  a single scalar shortcut. Implicitly assumes TCRE and ΔT are uncorrelated
  across configs, which need not hold.

Both methods are computed at each GTP timescale (20/50/100 yr → years
2040/2070/2120) and exported as quantile tables to `output/gtp_exact.csv` /
`output/gtp_shortcut.csv` (mean, 17%/83% = "likely", 2.5%/97.5% = "very
likely", matching IPCC calibrated-language percentile convention), plus
full time-series and per-member distribution comparisons in `figures/`
(`gtp_exact_vs_shortcut_{timeseries,hist,scatter}.png`,
`temperature_uncertainty.png`, `co2_equivalent_uncertainty.png`).

The script also prints order-of-magnitude sanity checks at
`example_year = 2070` (exact vs. shortcut GtCO2, dT mean/95% range) — keep
these when extending the script; they're the quick numeric gut-check for
any new unit conversion or aggregation.

## Conventions

- TCRE: K/GtCO2. Forcing: W/m². Cumulative/CO2-equivalent emissions: GtCO2.
- Uncertainty ranges follow IPCC calibrated language: 17th-83rd percentile
  = "likely" (66%), 2.5th-97.5th = "very likely" (95%).
- Temperatures are rebased to a 1850-1900 baseline (`rebase_temperature`)
  before comparison, matching IPCC convention.
