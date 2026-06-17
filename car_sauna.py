#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpers for running FaIR baseline and scaled activity pulses.

Created on Thu Mar 5 14:52:07 2026
@author: Antti-Ilari Partanen
"""

import warnings
from typing import Dict, Optional, Tuple

import xarray as xr

from fair_tools import createConstrainedRuns, rebase_temperature


def run_baseline_fair(
    base_scenario: str = 'ssp245',
    year_end: int = 2100,
    forcings: Optional[Dict[str, bool]] = None,
) -> Tuple[object, xr.DataArray]:
    """
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
            - 'timebounds'
            - 'config'
    """
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

    sat = f.temperature.sel(layer=0)
    drop_coords = [coord for coord in sat.coords if coord in ['scenario', 'layer']]
    if drop_coords:
        sat = sat.drop_vars(drop_coords)
    sat = sat.squeeze()

    return f, sat


def _apply_activity_pulse(
    f: object,
    year: float,
    activity_emissions: Dict[str, float],
    scale_factor: float,
) -> None:
    """
    Internal helper that modifies emissions in-place for one mid-year timepoint.

    Parameters
    ----------
    f : fair.fair.FAIR
        FaIR instance with emissions already set (from createConstrainedRuns).
    year : float
        Timepoint (e.g. 2026.5) at which to apply the emission pulse.
        Must match a value in f.emissions.timepoints.
    activity_emissions : dict
        Dictionary mapping specie name (as used in FaIR) to emissions
        per single unit activity (e.g. per sauna session), in the same
        units as the FaIR SSP emissions (e.g. GtCO2, Mt, etc.).
        Values are multiplied by 'scale_factor' before being added.
    scale_factor : float
        Scalar by which the activity_emissions are multiplied for the
        model run.
    """
    available_species = f.emissions.coords['specie'].values
    available_times = f.emissions.coords['timepoints'].values

    if year not in available_times:
        raise ValueError(
            f'Year {year} not in f.emissions.timepoints; '
            f'available range is {available_times.min()}–{available_times.max()}'
        )

    for specie, emis in activity_emissions.items():
        if specie not in available_species:
            raise ValueError(
                f'Specie "{specie}" not found in f.emissions. '
                f'Available species include: {list(available_species)}'
            )

        f.emissions.loc[dict(timepoints=year, specie=specie)] += emis * scale_factor


def run_scaled_activity_pulse(
    base_scenario: str = 'ssp245',
    year: int = 2026,
    activity_emissions: Dict[str, float] = None,
    activity_label: str = 'activity',
    year_end: int = 2100,
    scale_factor: float = 1e6,
    forcings: Optional[Dict[str, bool]] = None,
    f_base: Optional[object] = None,
    sat_base: Optional[xr.DataArray] = None,
    return_fair: bool = False,
):
    """
    Run FaIR for a baseline and a scaled activity pulse, return per-unit response.

    This function:
        1) uses a baseline FaIR run (given or newly run) for the chosen SSP;
        2) runs a perturbed simulation with emissions at mid-year 'year+0.5'
           increased by scale_factor * activity_emissions;
        3) computes per-unit temperature response by dividing the difference
           by 'scale_factor'.

    Parameters
    ----------
    base_scenario : str, optional
        SSP scenario name used as the background (e.g. 'ssp245').
        Default is 'ssp245'.
    year : int, optional
        Activity year (e.g. 2026). The pulse is applied at year+0.5.
        Default is 2026.
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
    f_base : fair.fair.FAIR, optional
        Pre-computed baseline FaIR instance (e.g. from run_baseline_fair).
        If provided, this baseline is reused and not re-run.
    sat_base : xarray.DataArray, optional
        Baseline rebased SAT corresponding to f_base. If None and f_base
        is provided, it is computed from f_base.temperature.
    return_fair : bool, optional
        If True, return also the FaIR perturbed instance. The return is
        then (ds, f_pert). Default is False.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing:
            - 'sat_baseline'      : baseline global SAT (rebased)
            - 'sat_perturbed'     : perturbed global SAT (rebased)
            - 'delta_sat_per_unit': (sat_perturbed - sat_baseline) / scale_factor
        All with dimensions:
            - 'timebounds'
            - 'config'
        And attributes:
            'activity', 'base_scenario', 'perturbation_year', 'scale_factor'.

    If return_fair is True:
        (ds, f_pert)
    """
    if activity_emissions is None:
        raise ValueError('activity_emissions must be provided.')

    if forcings is None:
        forcings = {'non-ghg': True, 'non-co2-ghgs': True}

    # 1. Baseline: reuse if provided, otherwise run a new one
    if f_base is None:
        f_base, sat_base_local = run_baseline_fair(
            base_scenario=base_scenario,
            year_end=year_end,
            forcings=forcings,
        )
        sat_base = sat_base_local
    else:
        if sat_base is None:
            sat_base = f_base.temperature.sel(layer=0)
            drop_coords = [
                coord for coord in sat_base.coords
                if coord in ['scenario', 'layer']
            ]
            if drop_coords:
                sat_base = sat_base.drop_vars(drop_coords)
            sat_base = sat_base.squeeze()

    # 2. Perturbed run: new FaIR instance to avoid side-effects
    f_pert = createConstrainedRuns(
        scenarios=[base_scenario],
        year_end=year_end,
        forcings=forcings,
    )

    _apply_activity_pulse(
        f=f_pert,
        year=year + 0.5,  # mid-year timepoint for emissions
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

    # 3. Per-unit temperature response (both on 'timebounds')
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

    if return_fair:
        return ds, f_pert

    return ds