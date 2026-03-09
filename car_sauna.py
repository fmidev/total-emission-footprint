#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:52:07 2026

@author: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

from fair_tools import createConstrainedRuns, rebase_temperature


def run_baseline_fair(
    base_scenario: str = 'ssp245',
    year_end: int = 2100,
    forcings: Optional[Dict[str, bool]] = None,
) -> Tuple[object, xr.DataArray]:
    '''
    Run a baseline FaIR simulation for a given SSP scenario.

    Parameters
    ----------
    base_scenario : str, optional
        Name of the SSP scenario to use, e.g. 'ssp245'.
        Must exist in the harmonized emissions NetCDF used by
        createConstrainedRuns. Default is 'ssp245'.
    year_end : int, optional
        Last simulation year. Must be >= 1750 and within the range
        of the input data. Default is 2100.
    forcings : dict, optional
        Dictionary controlling non-GHG and non-CO2 GHG options, e.g.
        {'non-ghg': True, 'non-co2-ghgs': True}.
        If None, the defaults of createConstrainedRuns are used.

    Returns
    -------
    f : fair.fair.FAIR
        The FaIR instance after running the baseline simulation.
    sat : xarray.DataArray
        Global mean surface air temperature (layer 0), rebased to
        1850–1900, with dimensions:
            - 'timepoints'
            - 'config'
    '''
    if forcings is None:
        forcings = {'non-ghg': True, 'non-co2-ghgs': True}

    f = createConstrainedRuns(
        scenarios=[base_scenario],
        year_end=year_end,
        forcings=forcings,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f.run(progress=False)

    f = rebase_temperature(f)

    # Select global-mean SAT (layer 0) and drop scenario/layer coords
    sat = f.temperature.sel(layer=0)
    # Remove scenario and layer coordinates if they exist
    drop_coords = [coord for coord in sat.coords if coord in ['scenario', 'layer']]
    if drop_coords:
        sat = sat.drop_vars(drop_coords)
    sat = sat.squeeze()

    return f, sat

def _apply_activity_pulse(
    f: object,
    year: int,
    activity_emissions: Dict[str, float],
    scale_factor: float,
) -> None:
    '''
    Internal helper that modifies emissions in-place for one year.

    Parameters
    ----------
    f : fair.fair.FAIR
        FaIR instance with emissions already set (from createConstrainedRuns).
    year : int
        Year in which to apply the emission pulse (must be in
        f.emissions.timepoints).
    activity_emissions : dict
        Dictionary mapping specie name (as used in FaIR) to emissions
        per single unit activity (e.g. per sauna session), in the same
        units as the FaIR SSP emissions (e.g. GtCO2, Mt, etc.).
        Values are multiplied by 'scale_factor' before being added.
    scale_factor : float
        Scalar by which the activity_emissions are multiplied for the
        model run.
    '''
    available_species = f.emissions.coords['specie'].values
    available_years = f.emissions.coords['timepoints'].values

    if year not in available_years:
        raise ValueError(
            f'Year {year} not in f.emissions.timepoints; '
            f'available range is {available_years.min()}–{available_years.max()}'
        )

    for specie, emis in activity_emissions.items():
        if specie not in available_species:
            raise ValueError(
                f'Specie "{specie}" not found in f.emissions. '
                f'Available species include: {list(available_species)}'
            )

        # Add scaled emissions to all configs and the single scenario.
        # Xarray will broadcast over 'config' and 'scenario' automatically.
        f.emissions.loc[dict(timepoints=year, specie=specie)] += emis * scale_factor


def run_scaled_activity_pulse(
    base_scenario: str = 'ssp245',
    year: int = 2026,
    activity_emissions: Dict[str, float] = None,
    activity_label: str = 'activity',
    year_end: int = 2100,
    scale_factor: float = 1e6,
    forcings: Optional[Dict[str, bool]] = None,
) -> xr.Dataset:
    '''
    Run FaIR for a baseline and a scaled activity pulse, return per-unit response.

    This function:
        1) runs a baseline simulation for the chosen SSP scenario;
        2) runs a perturbed simulation with emissions in 'year' increased by
           scale_factor * activity_emissions;
        3) computes per-unit temperature response by dividing the difference
           by 'scale_factor'.

    Parameters
    ----------
    base_scenario : str, optional
        SSP scenario name used as the background (e.g. 'ssp245').
        Default is 'ssp245'.
    year : int, optional
        Year in which the activity emissions are applied. Default is 2026.
    activity_emissions : dict
        Mapping from specie name (FaIR naming) to annual emissions per
        single unit of activity (e.g. one sauna activity). Units must be
        consistent with the FaIR emissions input.
    activity_label : str, optional
        Descriptive label for the activity (e.g. 'city_sauna',
        'cottage_sauna'). Stored as metadata in the output.
        Default is 'activity'.
    year_end : int, optional
        Last simulation year. Default is 2100.
    scale_factor : float, optional
        Factor by which 'activity_emissions' is multiplied in the model run.
        The climate response is divided by this factor to obtain per-unit
        effects. Default is 1e6.
    forcings : dict, optional
        Dictionary controlling non-GHG and non-CO2 GHG options, e.g.
        {'non-ghg': True, 'non-co2-ghgs': True}.
        If None, the defaults of createConstrainedRuns are used.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing:
            - 'sat_baseline'      : baseline global SAT (rebased)
            - 'sat_perturbed'     : perturbed global SAT (rebased)
            - 'delta_sat_per_unit': (sat_perturbed - sat_baseline) / scale_factor

        All with dimensions:
            - 'timepoints'
            - 'config'

        And attributes:
            'activity', 'base_scenario', 'perturbation_year', 'scale_factor'.
    '''
    if activity_emissions is None:
        raise ValueError('activity_emissions must be provided.')

    if forcings is None:
        forcings = {'non-ghg': True, 'non-co2-ghgs': True}

    # 1. Baseline run
    _, sat_base = run_baseline_fair(
        base_scenario=base_scenario,
        year_end=year_end,
        forcings=forcings,
    )

    # 2. Perturbed run: new FaIR instance to avoid side-effects
    f_pert = createConstrainedRuns(
        scenarios=[base_scenario],
        year_end=year_end,
        forcings=forcings,
    )

    _apply_activity_pulse(
        f=f_pert,
        year=year+0.5,
        activity_emissions=activity_emissions,
        scale_factor=scale_factor,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f_pert.run(progress=False)

    f_pert = rebase_temperature(f_pert)
    sat_pert = f_pert.temperature.sel(layer=0)
    drop_coords = [coord for coord in sat_pert.coords if coord in ['scenario', 'layer']]
    if drop_coords:
        sat_pert = sat_pert.drop_vars(drop_coords)
    sat_pert = sat_pert.squeeze()

    # 3. Per-unit temperature response
    delta_sat = (sat_pert - sat_base) / scale_factor

    # Add metadata
    sat_base = sat_base.assign_attrs(
        activity=activity_label,
        base_scenario=base_scenario,
        perturbation_year=year,
        scale_factor=scale_factor,
        run_type='baseline',
    )
    sat_pert = sat_pert.assign_attrs(
        activity=activity_label,
        base_scenario=base_scenario,
        perturbation_year=year,
        scale_factor=scale_factor,
        run_type='perturbed',
    )
    delta_sat = delta_sat.assign_attrs(
        activity=activity_label,
        base_scenario=base_scenario,
        perturbation_year=year,
        scale_factor=scale_factor,
        run_type='per_unit_response',
    )

    ds = xr.Dataset(
        {
            'sat_baseline': sat_base,
            'sat_perturbed': sat_pert,
            'delta_sat_per_unit': delta_sat,
        }
    )

    return ds


def run_activity_scenarios(
    base_scenario: str = 'ssp245',
    year: int = 2026,
    scenarios: Dict[str, Dict[str, float]] = None,
    year_end: int = 2100,
    scale_factor: float = 1e6,
    forcings: Optional[Dict[str, bool]] = None,
) -> Dict[str, xr.Dataset]:
    '''
    Run multiple activity scenarios and return per-unit responses.

    Parameters
    ----------
    base_scenario : str, optional
        SSP background scenario (e.g. 'ssp245'). Default is 'ssp245'.
    year : int, optional
        Year in which all activities are applied. Default is 2026.
    scenarios : dict
        Dictionary mapping scenario name to 'activity_emissions' dict, e.g.:
            {
                'city_sauna': {...},
                'cottage_sauna': {...},
            }
        Each inner dict maps specie -> emissions per unit activity.
    year_end : int, optional
        Last simulation year. Default is 2100.
    scale_factor : float, optional
        Scale factor for all scenarios. Default is 1e6.
    forcings : dict, optional
        Forcing options passed to createConstrainedRuns. If None, defaults
        of createConstrainedRuns are used.

    Returns
    -------
    results : dict
        Mapping from scenario name to xarray.Dataset as returned by
        run_scaled_activity_pulse.
    '''
    if scenarios is None:
        raise ValueError('scenarios dictionary must be provided.')

    results: Dict[str, xr.Dataset] = {}

    for name, activity_emissions in scenarios.items():
        ds = run_scaled_activity_pulse(
            base_scenario=base_scenario,
            year=year,
            activity_emissions=activity_emissions,
            activity_label=name,
            year_end=year_end,
            scale_factor=scale_factor,
            forcings=forcings,
        )
        results[name] = ds

    return results