#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:52:43 2026

@author: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
"""

# sauna_scenarios.py

from typing import Dict

from car_sauna import run_activity_scenarios


def g_to_gtco2(mass_g: float) -> float:
    '''
    Convert mass in grams to GtCO2 (gigatonnes CO2).

    Parameters
    ----------
    mass_g : float
        Mass in grams.

    Returns
    -------
    float
        Mass in gigatonnes CO2 (GtCO2).
    '''
    return mass_g / 1e15


def g_to_mt(mass_g: float) -> float:
    '''
    Convert mass in grams to Mt (megatonnes).

    Parameters
    ----------
    mass_g : float
        Mass in grams.

    Returns
    -------
    float
        Mass in megatonnes (Mt).
    '''
    return mass_g / 1e12


def get_sauna_emissions() -> Dict[str, float]:
    '''
    Return per-activity emissions for heating a wooden sauna, in FaIR units.

    All input values are defined in grams per sauna activity
    (e.g. per session or per defined "sauna use"), and converted to the
    units in the FaIR SSP file:

        - CO2 : GtCO2
        - BC  : Mt
        - OC  : Mt

    Replace the placeholder values with your actual estimates.

    Returns
    -------
    emissions : dict
        Species emissions per sauna activity, using FaIR SSP units.
    '''
    # TODO: replace these with your best estimates (grams per sauna activity)
    co2_sauna_g = 17e3
    bc_sauna_g = 86
    oc_sauna_g = 0
    voc_sauna_g= 290

    emissions = {
        'CO2': g_to_gtco2(co2_sauna_g),
        'BC': g_to_mt(bc_sauna_g),
        'OC': g_to_mt(oc_sauna_g),
        'VOC': g_to_mt(voc_sauna_g)
    }
    return emissions


def get_driving_emissions() -> Dict[str, float]:
    '''
    Return per-activity driving emissions for a round trip to the cottage,
    in FaIR units.

    Input values are grams per round trip, converted to:
        - CO2 : GtCO2
        - NOx : Mt
        - BC  : Mt

    Replace the placeholder values with your actual estimates.

    Returns
    -------
    emissions : dict
        Species emissions per round-trip drive, using FaIR SSP units.
    '''
    # TODO: replace these with your best estimates (grams per round trip)
    co2_drive_g = 90
    nox_drive_g = 11
    bc_drive_g = 0.19+0.075 # (combustion + road abrasion)
    voc_drive_g = 75

    emissions = {
        'CO2': g_to_gtco2(co2_drive_g),
        'NOx': g_to_mt(nox_drive_g),
        'BC': g_to_mt(bc_drive_g),
        'VOC': g_to_mt(voc_drive_g)
    }
    return emissions


# def main() -> None:
'''
Run FaIR for city and cottage sauna pulses in 2026 on top of ssp245,
using emissions defined in grams per activity, and print a simple
summary of the per-unit global temperature response.
'''
sauna = get_sauna_emissions()
drive = get_driving_emissions()

# City sauna: only sauna heating.
# Cottage sauna: sauna + driving.
all_species = set(sauna.keys()).union(drive.keys())
cottage_sauna = {
    specie: sauna.get(specie, 0.0) + drive.get(specie, 0.0)
    for specie in all_species
}

scenarios = {
    'city_sauna': sauna,
    'cottage_sauna': cottage_sauna,
}

results = run_activity_scenarios(
    base_scenario='ssp245',
    year=2026,
    scenarios=scenarios,
    year_end=2100,
    scale_factor=1e6,  # large pulse; response is divided by this
    forcings={'non-ghg': True, 'non-co2-ghgs': True},
)

# Example: median per-unit response at 2100 for each scenario.
for name, ds in results.items():
    delta_sat_median = ds['delta_sat_per_unit'].median(dim='config')
    value_2100 = float(delta_sat_median.sel(timebounds=2100))
    print(
        f'Scenario: {name:14s} | '
        f'ΔT per unit activity in 2100: {value_2100:.3e} K'
    )


# if __name__ == '__main__':
#     main()