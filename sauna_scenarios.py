#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sauna activity climate forcing study using FaIR.

Scenarios:
- city_sauna: only wooden sauna heating
- cottage_sauna: sauna heating + driving to cottage

Emissions inputs are in grams per activity and converted to FaIR units.
Baseline SSP245 is run once and reused for both perturbation experiments.
"""

from typing import Dict

import matplotlib.pyplot as plt

from car_sauna import run_baseline_fair, run_scaled_activity_pulse

import fair_tools


# -------------------------------------------------------------------
# Unit conversions
# -------------------------------------------------------------------

def g_to_gtco2(mass_g: float) -> float:
    """
    Convert mass in grams to GtCO2 (gigatonnes CO2).

    Parameters
    ----------
    mass_g : float
        Mass in grams.

    Returns
    -------
    float
        Mass in gigatonnes CO2 (GtCO2).
    """
    return mass_g / 1e15


def g_to_mt(mass_g: float) -> float:
    """
    Convert mass in grams to Mt (megatonnes).

    Parameters
    ----------
    mass_g : float
        Mass in grams.

    Returns
    -------
    float
        Mass in megatonnes (Mt).
    """
    return mass_g / 1e12


# -------------------------------------------------------------------
# Activity emissions
# -------------------------------------------------------------------

def get_sauna_emissions() -> Dict[str, float]:
    """
    Return per-activity emissions for heating a wooden sauna, in FaIR units.

    Input values are defined in grams per sauna activity and converted to:
        - CO2 FFI : GtCO2
        - BC      : Mt
        - OC      : Mt
        - VOC     : Mt

    Returns
    -------
    emissions : dict
        Species emissions per sauna activity, using FaIR SSP units.
    """
    co2_sauna_g = 17e3
    bc_sauna_g = 86
    oc_sauna_g = 0
    voc_sauna_g = 290

    emissions = {
        'CO2 FFI': g_to_gtco2(co2_sauna_g),
        'BC': g_to_mt(bc_sauna_g),
        'OC': g_to_mt(oc_sauna_g),
        'VOC': g_to_mt(voc_sauna_g),
    }
    return emissions


def get_driving_emissions() -> Dict[str, float]:
    """
    Return per-activity driving emissions for a round trip to the cottage,
    in FaIR units.

    Input values are grams per round trip, converted to:
        - CO2 FFI : GtCO2
        - NOx     : Mt
        - BC      : Mt
        - VOC     : Mt

    Returns
    -------
    emissions : dict
        Species emissions per round-trip drive, using FaIR SSP units.
    """
    co2_drive_g = 90
    nox_drive_g = 11
    bc_drive_g = 0.19 + 0.075  # combustion + road abrasion
    voc_drive_g = 75

    emissions = {
        'CO2 FFI': g_to_gtco2(co2_drive_g),
        'NOx': g_to_mt(nox_drive_g),
        'BC': g_to_mt(bc_drive_g),
        'VOC': g_to_mt(voc_drive_g),
    }
    return emissions


# -------------------------------------------------------------------
# RUN BASELINE ONCE, THEN TWO PERTURBATIONS
# -------------------------------------------------------------------

BASE_SCENARIO = 'ssp245'
ACTIVITY_YEAR = 2026
YEAR_END = 2100
SCALE_FACTOR = 1e6
FORCINGS = {'non-ghg': True, 'non-co2-ghgs': True}

# Baseline FaIR run (shared for all perturbations)
f_base, sat_base = run_baseline_fair(
    base_scenario=BASE_SCENARIO,
    year_end=YEAR_END,
    forcings=FORCINGS,
)

# Scenario-specific emissions in FaIR units
sauna = get_sauna_emissions()
drive = get_driving_emissions()

all_species = set(sauna.keys()).union(drive.keys())
cottage_sauna = {
    specie: sauna.get(specie, 0.0) + drive.get(specie, 0.0)
    for specie in all_species
}

# City sauna perturbation
ds_city, f_pert_city = run_scaled_activity_pulse(
    base_scenario=BASE_SCENARIO,
    year=ACTIVITY_YEAR,
    activity_emissions=sauna,
    activity_label='city_sauna',
    year_end=YEAR_END,
    scale_factor=SCALE_FACTOR,
    forcings=FORCINGS,
    f_base=f_base,
    return_fair=True,
)

# Cottage sauna perturbation
ds_cottage, f_pert_cottage = run_scaled_activity_pulse(
    base_scenario=BASE_SCENARIO,
    year=ACTIVITY_YEAR,
    activity_emissions=cottage_sauna,
    activity_label='cottage_sauna',
    year_end=YEAR_END,
    scale_factor=SCALE_FACTOR,
    forcings=FORCINGS,
    f_base=f_base,
    return_fair=True,
)

# Extract per-unit ΔT time series (median across configs)
delta_city = ds_city['delta_sat_per_unit']
delta_cottage = ds_cottage['delta_sat_per_unit']

delta_city_median = delta_city.median(dim='config')
delta_cottage_median = delta_cottage.median(dim='config')


# -------------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------------

def plot_temperature_differences() -> None:
    """
    Plot ΔT(time) per unit activity for city and cottage sauna scenarios.

    Produces two figures:
        1) ΔT_city(time)
        2) ΔT_cottage(time)
    """
    # City sauna
    plt.figure(figsize=(10, 5))
    plt.plot(
        delta_city_median['timebounds'],
        delta_city_median,
        label='City sauna (per activity)',
    )
    plt.xlabel('Year')
    plt.ylabel('ΔT [K]')
    plt.title('Temperature response per city sauna activity')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Cottage sauna
    plt.figure(figsize=(10, 5))
    plt.plot(
        delta_cottage_median['timebounds'],
        delta_cottage_median,
        label='Cottage sauna (per activity)',
        color='orange',
    )
    plt.xlabel('Year')
    plt.ylabel('ΔT [K]')
    plt.title('Temperature response per cottage sauna activity')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_temperature_differences()