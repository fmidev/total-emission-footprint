# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 07:22:08 2023

@author: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)

FaIR 2.2.4 / calibration v1.6.0. Ported from the ghgbudgets repo's
fair_runs.py (createConstrainedRuns, rebase_temperature, calculate_timemean,
_clip_emissions_to_baseline) and from livestock_methane's script_00_tcre.py
(compute_tcre, lookup-based TCRE from the calibration's 1pctCO2 diagnostics).
"""
import numpy as np
import pandas as pd
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise
from dotenv import load_dotenv
import os
import xarray as xr
from pathlib import Path
import warnings


load_dotenv()

output_ensemble_size = int(os.getenv('POSTERIOR_SAMPLES'))

fair_calibration_dir = Path(os.getenv('FAIR_CALIBRATION_DIR'))


# Should 'Equivalent effective stratospheric chlorine' be included here?
non_ghg_forcings=['Solar', 'Volcanic', 'Land use']

non_ghg_species=['Sulfur', 'BC', 'OC',
       'NH3', 'NOx', 'VOC', 'CO']

non_co2_ghgs=['CH4', 'N2O', 'CFC-11', 'CFC-12', 'CFC-113', 'CFC-114',
       'CFC-115', 'HCFC-22', 'HCFC-141b', 'HCFC-142b', 'CCl4', 'CHCl3',
       'CH2Cl2', 'CH3Cl', 'CH3CCl3', 'CH3Br', 'Halon-1211', 'Halon-1301',
       'Halon-2402', 'CF4', 'C2F6', 'C3F8', 'c-C4F8', 'C4F10', 'C5F12',
       'C6F14', 'C7F16', 'C8F18', 'NF3', 'SF6', 'SO2F2', 'HFC-125', 'HFC-134a',
       'HFC-143a', 'HFC-152a', 'HFC-227ea', 'HFC-23', 'HFC-236fa', 'HFC-245fa',
       'HFC-32', 'HFC-365mfc', 'HFC-4310mee']


def _clip_emissions_to_baseline(f, verbose=False):
    """Clip non-CO2 GHG emissions to >= baseline_emissions in all scenarios.

    Physical floor: step_concentration uses (emissions - baseline_emissions)
    as the anthropogenic perturbation. Anything below baseline_emissions implies
    human activity is suppressing natural sources, which is unphysical.

    For F-gases baseline_emissions = 0, so this is a zero floor.
    For CH4/N2O the floor is the pre-industrial natural emission rate.

    The main practical trigger is CFC-115: the harmonized SSP file contains
    negative values from 2024 onward as a harmonization artifact (RCMIP v5.1.0
    assumes ~1.4 kt/yr at 2020; the CMIP7 historical record shows ~0 kt/yr by
    2021-2022; aneris pulls the future trajectory below zero to reconcile them).
    """
    for sp in non_co2_ghgs:
        floor = float(
            f.species_configs['baseline_emissions'].sel(specie=sp).mean()
        )
        vals = f.emissions.sel(specie=sp)
        below_floor = vals < floor
        n_years = int(below_floor.any(dim=['scenario', 'config']).sum())
        if n_years > 0:
            if verbose:
                min_val = float(vals.where(below_floor).min())
                scen_names = [str(s) for s in f.emissions.scenario.values]
                print(f'  Clipping {sp}: {n_years} year(s) below baseline floor '
                      f'({floor:.4g}), min value {min_val:.4g}, set to floor '
                      f'(scenarios: {scen_names})')
            f.emissions.loc[dict(specie=sp)] = vals.clip(min=floor)


def calculate_timemean(data_in, timebound_interval):
    tb1, tb2 = timebound_interval

    # number of time bounds: tb1 and tb2 are year-start integers, so +2 gives the
    # count of annual endpoints (tb1, tb1+1, …, tb2, tb2+1)
    n_tb = tb2 - tb1 + 2

    # trapezoidal weights: 0.5 at ends, 1.0 in middle
    weights = np.ones(n_tb)
    weights[0] = weights[-1] = 0.5
    weights /= weights.sum()

    # select interval, note tb2+1 to include closing bound
    data_sel = data_in.sel(timebounds=slice(tb1, tb2+1))

    # align weights to timebounds
    weight_da = xr.DataArray(weights, dims=['timebounds'], coords={'timebounds': data_sel['timebounds']})

    # weighted mean
    return (data_sel * weight_da).sum(dim='timebounds')


def createConstrainedRuns(scenarios=['ssp119'], year_end=2051, forcings={'non-ghg':True, 'non-co2-ghgs':True}, verbose=False):
    '''
    Based on the script:
        fair-calibrate/input/constraining/07_constrained-ssp-projections.py
        (calibration v1.6.0, FaIR 2.2.4)

    Parameters
    ----------
    scenarios : list of str
        SSP scenario names present in the harmonized emissions NetCDF.

    Returns
    -------
    f : FAIR
        Configured but not yet run.
    '''

    # Solar ERF: relative to 1850-2019 baseline (consistent with v1.6.0 calibration)
    df_solar = pd.read_csv(
        fair_calibration_dir / 'output' / 'forcing' / 'solar_forcing_timebounds_cmip7.csv',
        index_col=0,
    )
    # Volcanic ERF: relative to 1850-2021 baseline
    df_volcanic = pd.read_csv(
        fair_calibration_dir / 'data' / 'forcing' / 'volcanic_forcing_timebounds_cmip7.csv',
        index_col=0,
    )

    nyears = year_end - 1750 + 1

    solar_forcing = df_solar['solar_erf_rel_1850-2019'].loc[1750:year_end].values
    volcanic_forcing = np.zeros(nyears)
    year_end_volcanic = min(2301, year_end)
    volcanic_forcing[:year_end_volcanic - 1750 + 1] = (
        df_volcanic['volcanic_erf_rel_1850-2021'].loc[1750:year_end_volcanic].values
    )

    df_configs = pd.read_csv(
        fair_calibration_dir / 'output' / 'posteriors' / 'calibrated_constrained_parameters.csv',
        index_col=0,
    )

    valid_all = df_configs.index

    f = FAIR(ch4_method='Thornhill2021')
    f.define_time(1750, year_end, 1)
    f.define_scenarios(scenarios)
    f.define_configs(valid_all)

    # FaIR 2.2: read_properties takes path to calibration-specific species CSV
    species, properties = read_properties(
        fair_calibration_dir / 'output' / 'posteriors' / 'species_configs_properties.csv'
    )
    species.remove('Irrigation')
    # Land use forcing is derived from CO2 AFOLU cumulative emissions when active;
    # switched to prescribed (filled with 0) when non-ghg forcings are suppressed.
    if forcings['non-ghg']:
        properties['Land use']['input_mode'] = 'calculated'
    else:
        properties['Land use']['input_mode'] = 'forcing'

    f.define_species(species, properties)
    f.allocate()

    da_emissions = xr.load_dataarray(
        fair_calibration_dir / 'output' / 'emissions' / 'ssps_harmonized_1750-2499.nc'
    )

    da = da_emissions.loc[dict(config='unspecified', scenario=scenarios, specie=species)][:nyears-1, ...]
    fe = da.expand_dims(dim=['config'], axis=(2))
    f.emissions = fe.drop_vars('config') * np.ones((1, 1, output_ensemble_size, 1))
    f.emissions.coords['config'] = f.configs

    if forcings['non-ghg']:
        fill(
            f.forcing,
            volcanic_forcing[:, None, None] * df_configs['forcing_scale[Volcanic]'].values.squeeze(),
            specie='Volcanic',
        )
        fill(
            f.forcing,
            solar_forcing[:, None, None] * df_configs['forcing_scale[Solar]'].values.squeeze(),
            specie='Solar',
        )
    else:
        fill(f.forcing, 0., specie='Volcanic')
        fill(f.forcing, 0., specie='Solar')
        fill(f.forcing, 0., specie='Land use')

    # FaIR 2.2: fill species defaults from calibration properties, then override
    # with the full posterior parameter set (climate response, carbon cycle,
    # aerosols, ozone, per-species forcing scales, etc.).  baseline_emissions,
    # land_use_cumulative_emissions_to_forcing, and lapsi_radiative_efficiency
    # all live in species_configs_properties.csv for v1.6.0 — no manual fill
    # needed (matches reference 07_constrained-ssp-projections.py).
    f.fill_species_configs(
        fair_calibration_dir / 'output' / 'posteriors' / 'species_configs_properties.csv'
    )
    f.override_defaults(
        fair_calibration_dir / 'output' / 'posteriors' / 'calibrated_constrained_parameters.csv'
    )

    if not forcings['non-ghg']:
        for specie in non_ghg_species:
            f.emissions.loc[dict(specie=specie)] = (
                f.species_configs['baseline_emissions'].loc[dict(specie=specie)])
    if not forcings['non-co2-ghgs']:
        for specie in non_co2_ghgs:
            f.emissions.loc[dict(specie=specie)] = (
                f.species_configs['baseline_emissions'].loc[dict(specie=specie)])

    # Enforce physical floor: clip non-CO2 GHG emissions to >= baseline_emissions.
    # Species_configs are fully populated above, so baseline values are available.
    # The non-co2-ghgs override above (if active) already sets emissions to the
    # baseline floor, so this call will be a no-op in that case.
    _clip_emissions_to_baseline(f, verbose=verbose)

    initialise(f.concentration, f.species_configs['baseline_concentration'])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    return f


# Readjust temperatures to be relative to 1850-1900
def rebase_temperature(f):
   f.temperature = f.temperature - calculate_timemean(f.temperature, [1850, 1900])
   return f


def update_scenario_names(f, scenario_map):
    """
    Update scenario names in all DataArrays of a Fair instance and in the 'scenarios' attribute.

    Parameters:
    f : fair.fair.FAIR
        The Fair instance containing DataArrays to update.
    scenario_map : dict
        A dictionary where keys are current scenario names and values are the new names.

    Returns:
    f : fair.fair.FAIR
        The modified Fair instance with updated scenario names.
    """
    # Update scenarios attribute if it exists
    if hasattr(f, 'scenarios'):
        f.scenarios = [scenario_map.get(scenario, scenario) for scenario in f.scenarios]

    # Update scenario coordinates in all DataArrays
    for attr in dir(f):
        data_array = getattr(f, attr)
        if isinstance(data_array, xr.DataArray) and 'scenario' in data_array.coords:
            # Replace scenario names based on scenario_map
            current_scenarios = data_array.coords['scenario'].values
            new_scenarios = [scenario_map.get(s, s) for s in current_scenarios]
            data_array = data_array.assign_coords(scenario=new_scenarios)
            # Reassign the updated DataArray back to the Fair instance attribute
            setattr(f, attr, data_array)

    return f


def compute_tcre():
    '''
    Look up TCRE (and a temperature-at-1000-GtC diagnostic) from the FaIR
    calibration archive's own 1pctCO2 experiment, instead of re-running it.

    Based on the script:
        livestock_methane/script_00_tcre.py

    The calibration archive ran a 1pctCO2 FaIR experiment once for all 1.6M
    prior configs (Smith et al. 2024 calibration) and stored scalar
    diagnostics in output/prior_runs/. TCRE is derived as TCR / 3670 GtCO2
    (3670 Gt CO2 = 1000 Gt C, the standard cumulative CO2 at CO2 doubling,
    year 70 of the 1%/yr experiment); this is the same approximation used by
    IPCC AR6 (ignores per-config airborne-fraction variation).

    Returns
    -------
    tcre : xarray.DataArray
        Transient Climate Response to Cumulative CO2 Emissions, K/GtCO2,
        dim 'config' (coords = the 841 constrained config IDs).
    sat_1000gtc : xarray.DataArray
        Temperature (K) at 1000 Gt C cumulative CO2 emissions in the
        1pctCO2 experiment, same 'config' dim/coords as tcre. Useful as a
        single-point illustration of the TCRE relationship (the calibration
        archive only stores this scalar snapshot, not the full year-by-year
        trajectory for the constrained ensemble).
    '''
    # 1000 Gt C of cumulative CO2 at CO2 doubling, converted to Gt CO2.
    cumulative_co2_at_doubling = 3670.0  # Gt CO2

    df_configs = pd.read_csv(
        fair_calibration_dir / 'output' / 'posteriors' / 'calibrated_constrained_parameters.csv',
        index_col=0,
    )
    config_ids = df_configs.index

    # tcr.npy covers all 1.6M prior runs and is indexed directly by raw prior
    # config ID, so the reweighted-pass run IDs (== config_ids) index into it
    # directly.
    tcr_all = np.load(fair_calibration_dir / 'output' / 'prior_runs' / 'tcr.npy')
    run_ids = pd.read_csv(
        fair_calibration_dir / 'output' / 'posteriors' / 'runids_rmse_reweighted_pass.csv',
        header=None,
    )[0].values
    tcr_constrained = tcr_all[run_ids]
    tcre = xr.DataArray(
        tcr_constrained / cumulative_co2_at_doubling,
        dims=['config'],
        coords={'config': config_ids},
    )

    # temperature_1pctCO2_1000GtC.npy is only available for the RMSE-pass
    # subset of priors, and is indexed by *position within*
    # runids_rmse_pass.csv, not by raw config ID. Build a position lookup
    # (rather than np.isin(...).nonzero(), used for printing only in the
    # calibration's own check script) so the result stays ordered to match
    # config_ids / the 'config' coordinate used everywhere else.
    t1000_all = np.load(
        fair_calibration_dir / 'output' / 'prior_runs' / 'temperature_1pctCO2_1000GtC.npy'
    )
    rmse_pass_ids = np.loadtxt(
        fair_calibration_dir / 'output' / 'posteriors' / 'runids_rmse_pass.csv',
        dtype=int,
    )
    position_in_rmse_pass = {cid: pos for pos, cid in enumerate(rmse_pass_ids)}
    idx = np.array([position_in_rmse_pass[cid] for cid in config_ids])
    sat_1000gtc = xr.DataArray(
        t1000_all[idx],
        dims=['config'],
        coords={'config': config_ids},
    )

    return tcre, sat_1000gtc
