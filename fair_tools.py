# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 07:22:08 2023

@author: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
"""
import numpy as np
import pandas as pd
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise
from dotenv import load_dotenv
import os
import netCDF4
import xarray as xr
from pathlib import Path


load_dotenv()

cal_v = os.getenv('CALIBRATION_VERSION')
fair_v = os.getenv('FAIR_VERSION')
constraint_set = os.getenv('CONSTRAINT_SET')
output_ensemble_size = int(os.getenv('POSTERIOR_SAMPLES'))
plots = os.getenv('PLOTS', 'False').lower() in ('true', '1', 't')


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


def createConstrainedRuns(scenarios=['ssp119'], year_end=2051, forcings={'non-ghg':True, 'non-co2-ghgs':True}):
    '''
    Based on the script:
        fair-calibrate/input/fair-2.1.3/v1.4/all-2022/constraining/05_constrained-ssp-projections.py 
        (from 5a31144b58d6b9e2b23a845f68f73c7eda98701c)
    
    Parameters
    ----------
    scenarios : TYPE, optional
        DESCRIPTION. The default is ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585'].
    
    Returns
    -------
    f : TYPE
        DESCRIPTION.
    
    '''
    datadir=Path('/Users/partanen/OneDrive - Ilmatieteen laitos/projects/ghgbudgets/data')
    fair_calibration_dir=datadir / 'fair_calibrate'
    #fair_calibration_dir=Path('/Users/partanen/Library/CloudStorage/OneDrive-Ilmatieteenlaitos/projects/FaIR-MCMC/fair-calibrate')
    
    df_solar = pd.read_csv(
   fair_calibration_dir / 'data/forcing/solar_erf_timebounds.csv', index_col='year'
    )
    df_volcanic = pd.read_csv(
        fair_calibration_dir / 'data/forcing/volcanic_ERF_1750-2101_timebounds.csv',
        index_col='timebounds',
    )
    
    
    nyears=year_end-1750+1
    

    solar_forcing = np.zeros(nyears)
    volcanic_forcing = np.zeros(nyears)
    year_end_volcanic=min(2101,year_end)
    volcanic_forcing[:year_end_volcanic-1750+1] = df_volcanic['erf'].loc[1750:year_end_volcanic].values
    solar_forcing = df_solar['erf'].loc[1750:year_end].values
    
    df_methane = pd.read_csv(
        f'{fair_calibration_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/'
        'CH4_lifetime.csv',
        index_col=0,
    )
    df_configs = pd.read_csv(
        f'{fair_calibration_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/'
        'calibrated_constrained_parameters.csv',
        index_col=0,
    )
    
    df_landuse = pd.read_csv(
        f'{fair_calibration_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/'
        'landuse_scale_factor.csv',
        index_col=0,
    )
    df_lapsi = pd.read_csv(
        f'{fair_calibration_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/'
        'lapsi_scale_factor.csv',
        index_col=0,
    )
    valid_all = df_configs.index
    
    trend_shape = np.ones(nyears)
    trend_shape[:271] = np.linspace(0, 1, 271)
    
    f = FAIR(ch4_method='Thornhill2021')
    f.define_time(1750, year_end, 1)
    f.define_scenarios(scenarios)
    f.define_configs(valid_all)
    species, properties = read_properties()
    species.remove('Halon-1202')
    species.remove('NOx aviation')
    species.remove('Contrails')
    
# if not 'non-CO2-ghgs' in forcings:
#     for specie in non_co2_ghg_forcings:
#         species.remove(specie)
    
    # if not forcings['non-ghg']:
    #     for specie in non_ghg_forcings:
    #         species.remove(specie)
    
    f.define_species(species, properties)
    f.allocate()
    
    # run with harmonized emissions
    da_emissions = xr.load_dataarray(
        f'{fair_calibration_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/'
        'ssps_harmonized_1750-2499.nc'
    )
    
    da = da_emissions.loc[dict(config='unspecified', scenario=scenarios, specie=species)][:nyears-1, ...]
    fe = da.expand_dims(dim=['config'], axis=(2))
    f.emissions = fe.drop_vars('config') * np.ones((1, 1, output_ensemble_size, 1))
    f.emissions.coords['config'] = f.configs
    
    
    if forcings['non-ghg']:
        # solar and volcanic forcing
        fill(
            f.forcing,
            volcanic_forcing[:, None, None] * df_configs['fscale_Volcanic'].values.squeeze(),
            specie='Volcanic',
        )
        fill(
            f.forcing,
            solar_forcing[:, None, None] * df_configs['fscale_solar_amplitude'].values.squeeze()
            + trend_shape[:, None, None] * df_configs['fscale_solar_trend'].values.squeeze(),
            specie='Solar',
        )
    else:
        # solar and volcanic forcing
        fill(
            f.forcing,
            0.,
            specie='Volcanic',
        )
        fill(
            f.forcing,
            0.,
            specie='Solar',
        )
    
    # climate response
    fill(
        f.climate_configs['ocean_heat_capacity'],
        df_configs.loc[:, 'clim_c1':'clim_c3'].values,
    )
    fill(
        f.climate_configs['ocean_heat_transfer'],
        df_configs.loc[:, 'clim_kappa1':'clim_kappa3'].values,
    )  # not massively robust, since relies on kappa1, kappa2, kappa3 being in adjacent cols
    fill(
        f.climate_configs['deep_ocean_efficacy'],
        df_configs['clim_epsilon'].values.squeeze(),
    )
    fill(
        f.climate_configs['gamma_autocorrelation'],
        df_configs['clim_gamma'].values.squeeze(),
    )
    fill(f.climate_configs['sigma_eta'], df_configs['clim_sigma_eta'].values.squeeze())
    fill(f.climate_configs['sigma_xi'], df_configs['clim_sigma_xi'].values.squeeze())
    fill(f.climate_configs['seed'], df_configs['seed'])
    fill(f.climate_configs['stochastic_run'], True)
    fill(f.climate_configs['use_seed'], True)
    fill(f.climate_configs['forcing_4co2'], df_configs['clim_F_4xCO2'])
    
    # species level
    f.fill_species_configs()
    
    # carbon cycle
    fill(f.species_configs['iirf_0'], df_configs['cc_r0'].values.squeeze(), specie='CO2')
    fill(
        f.species_configs['iirf_airborne'],
        df_configs['cc_rA'].values.squeeze(),
        specie='CO2',
    )
    fill(
        f.species_configs['iirf_uptake'], df_configs['cc_rU'].values.squeeze(), specie='CO2'
    )
    fill(
        f.species_configs['iirf_temperature'],
        df_configs['cc_rT'].values.squeeze(),
        specie='CO2',
    )
    
    # correct land use scale factor term
    fill(
        f.species_configs['land_use_cumulative_emissions_to_forcing'],
        df_landuse.loc['historical_best', 'CO2_AFOLU'],
        specie='CO2 AFOLU',
    )
    
    
    # aerosol indirect
    fill(f.species_configs['aci_scale'], df_configs['aci_beta'].values.squeeze())
    fill(
        f.species_configs['aci_shape'],
        df_configs['aci_shape_so2'].values.squeeze(),
        specie='Sulfur',
    )
    fill(
        f.species_configs['aci_shape'],
        df_configs['aci_shape_bc'].values.squeeze(),
        specie='BC',
    )
    fill(
        f.species_configs['aci_shape'],
        df_configs['aci_shape_oc'].values.squeeze(),
        specie='OC',
    )
    
    # methane lifetime baseline and sensitivity
    fill(
        f.species_configs['unperturbed_lifetime'],
        df_methane.loc['historical_best', 'base'],
        specie='CH4',
    )
    fill(
        f.species_configs['ch4_lifetime_chemical_sensitivity'],
        df_methane.loc['historical_best', 'CH4'],
        specie='CH4',
    )
    fill(
        f.species_configs['ch4_lifetime_chemical_sensitivity'],
        df_methane.loc['historical_best', 'N2O'],
        specie='N2O',
    )
    fill(
        f.species_configs['lifetime_temperature_sensitivity'],
        df_methane.loc['historical_best', 'temp'],
    )
    
        

    fill(
        f.species_configs['ch4_lifetime_chemical_sensitivity'],
        df_methane.loc['historical_best', 'VOC'],
        specie='VOC',
    )
    fill(
        f.species_configs['ch4_lifetime_chemical_sensitivity'],
        df_methane.loc['historical_best', 'NOx'],
        specie='NOx',
    )
    fill(
        f.species_configs['ch4_lifetime_chemical_sensitivity'],
        df_methane.loc['historical_best', 'HC'],
        specie='Equivalent effective stratospheric chlorine',
    )
    # correct LAPSI scale factor term
    fill(
        f.species_configs['lapsi_radiative_efficiency'],
        df_lapsi.loc['historical_best', 'BC'],
        specie='BC',
    )
            
    
    
    # emissions adjustments for N2O and CH4 (we don't want to make these defaults as people
    # might wanna run pulse expts with these gases)
    fill(f.species_configs['baseline_emissions'], 19.41683292, specie='NOx')
    fill(f.species_configs['baseline_emissions'], 2.293964929, specie='Sulfur')
    fill(f.species_configs['baseline_emissions'], 348.4549732, specie='CO')
    fill(f.species_configs['baseline_emissions'], 60.62284009, specie='VOC')
    fill(f.species_configs['baseline_emissions'], 2.096765609, specie='BC')
    fill(f.species_configs['baseline_emissions'], 15.44571911, specie='OC')
    fill(f.species_configs['baseline_emissions'], 6.656462698, specie='NH3')
    fill(f.species_configs['baseline_emissions'], 38.246272, specie='CH4')
    fill(f.species_configs['baseline_emissions'], 0.92661989, specie='N2O')
    fill(f.species_configs['baseline_emissions'], 0.02129917, specie='CCl4')
    fill(f.species_configs['baseline_emissions'], 202.7251231, specie='CHCl3')
    fill(f.species_configs['baseline_emissions'], 211.0095537, specie='CH2Cl2')
    fill(f.species_configs['baseline_emissions'], 4544.519056, specie='CH3Cl')
    fill(f.species_configs['baseline_emissions'], 111.4920237, specie='CH3Br')
    fill(f.species_configs['baseline_emissions'], 0.008146006, specie='Halon-1211')
    fill(f.species_configs['baseline_emissions'], 0.000010554155, specie='SO2F2')
    fill(f.species_configs['baseline_emissions'], 0, specie='CF4')
    

    # aerosol direct
    for specie in [
        'BC',
        'CH4',
        'N2O',
        'NH3',
        'NOx',
        'OC',
        'Sulfur',
        'VOC',
        'Equivalent effective stratospheric chlorine',
    ]:
        fill(
            f.species_configs['erfari_radiative_efficiency'],
            df_configs[f'ari_{specie}'],
            specie=specie,
        )
    
    # forcing scaling
    fill(
        f.species_configs['forcing_scale'],
        df_configs['fscale_CO2'].values.squeeze(),
        specie='CO2',
    )

    for specie in [
        'CH4',
        'N2O'
    ]:
        fill(
            f.species_configs['forcing_scale'],
            df_configs[f'fscale_{specie}'].values.squeeze(),
            specie=specie,
        )

    for specie in [
        'Stratospheric water vapour',
        'Light absorbing particles on snow and ice'

    ]:
        fill(
            f.species_configs['forcing_scale'],
            df_configs[f'fscale_{specie}'].values.squeeze(),
            specie=specie,
        )
    if forcings['non-ghg']:
        fill(
            f.species_configs['forcing_scale'],
            df_configs['fscale_Land use'].values.squeeze(),
            specie='Land use',
        )
    

    for specie in [
        'CFC-11',
        'CFC-12',
        'CFC-113',
        'CFC-114',
        'CFC-115',
        'HCFC-22',
        'HCFC-141b',
        'HCFC-142b',
        'CCl4',
        'CHCl3',
        'CH2Cl2',
        'CH3Cl',
        'CH3CCl3',
        'CH3Br',
        'Halon-1211',
        'Halon-1301',
        'Halon-2402',
        'CF4',
        'C2F6',
        'C3F8',
        'c-C4F8',
        'C4F10',
        'C5F12',
        'C6F14',
        'C7F16',
        'C8F18',
        'NF3',
        'SF6',
        'SO2F2',
        'HFC-125',
        'HFC-134a',
        'HFC-143a',
        'HFC-152a',
        'HFC-227ea',
        'HFC-23',
        'HFC-236fa',
        'HFC-245fa',
        'HFC-32',
        'HFC-365mfc',
        'HFC-4310mee',
    ]:
        fill(
            f.species_configs['forcing_scale'],
            df_configs['fscale_minorGHG'].values.squeeze(),
            specie=specie,
        )

    # ozone

    for specie in [
        'Equivalent effective stratospheric chlorine',
        'CO',
        'VOC',
        'NOx',
    ]:
        fill(
            f.species_configs['ozone_radiative_efficiency'],
            df_configs[f'o3_{specie}'],
            specie=specie,
        )
    for specie in [
        'CH4',
        'N2O'
    ]:
        fill(
            f.species_configs['ozone_radiative_efficiency'],
            df_configs[f'o3_{specie}'],
            specie=specie,
        )
    
    if forcings['non-ghg']:
        # tune down volcanic efficacy
        fill(f.species_configs['forcing_efficacy'], 0.6, specie='Volcanic')
    
    
    # initial condition of CO2 concentration (but not baseline for forcing calculations)
    fill(
        f.species_configs['baseline_concentration'],
        df_configs['cc_co2_concentration_1750'].values.squeeze(),
        specie='CO2',
    )
    
    if not forcings['non-ghg']:
        # Overwrite emissions for non-GHG emissions with baseline emissions
        for specie in non_ghg_species:
            f.emissions.loc[dict(specie=specie)]=f.species_configs['baseline_emissions'].loc[dict(specie=specie)]
    
    if not forcings['non-co2-ghgs']:
        # Overwrite emissions for non-CO2 GHG emissions with baseline emissions
        for specie in non_co2_ghgs:
            f.emissions.loc[dict(specie=specie)]=f.species_configs['baseline_emissions'].loc[dict(specie=specie)]
    
    # initial conditions
    initialise(f.concentration, f.species_configs['baseline_concentration'])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    
    return f




# Readjust temperatures to be relative to 1850-1900
def rebase_temperature(f):
   f.temperature=f.temperature-f.temperature.sel(timebounds=slice(1850,1900)).mean(dim='timebounds') 
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

def calculate_natural_sinks(fair_instance, specie='CO2', reference_year=2024):
    """
    Calculate annual natural sinks for all scenarios in a given specie
    based on cumulative emissions and airborne fraction.

    Parameters:
    fair_instance : xarray.Dataset
        The FaIR instance containing airborne fraction and cumulative emissions.
    specie : str, optional
        The specie (e.g., 'CO2') for which to calculate natural sinks. Default is 'CO2'.

    Returns:
    annual_natural_sinks : xarray.DataArray
        The annual natural sinks calculated by differencing cumulative sinks for all scenarios.
    """
    # Select cumulative emissions and airborne fraction for the specified specie across all scenarios
    airborne_fraction = fair_instance.airborne_fraction.sel(specie=specie)
    cumulative_emissions = fair_instance.cumulative_emissions.sel(specie=specie)


    # Calculate cumulative natural sinks for all scenarios
    cumulative_natural_sinks = cumulative_emissions * (1 - airborne_fraction)

    # Compute annual natural sinks by differencing the cumulative values along the timebounds dimension
    annual_natural_sinks = cumulative_natural_sinks.diff(dim='timebounds')

    # Drop NaN values (for the first time bound)
    annual_natural_sinks = -annual_natural_sinks.dropna(dim='timebounds', how='all')
    cumulative_natural_sinks = -cumulative_natural_sinks.dropna(dim='timebounds', how='all')
    
    # Adjust cumulative natural sinks to zero at reference year
    if reference_year is not None:
        cumulative_natural_sinks=cumulative_natural_sinks-cumulative_natural_sinks.sel(timebounds=reference_year)
        


    return annual_natural_sinks, cumulative_natural_sinks
