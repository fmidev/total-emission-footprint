#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 09:58:11 2025

@author: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
"""

import fair_tools
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from pathlib import Path

year_end=2121

figpath=Path('figures')

ssp='ssp245'

f_base=fair_tools.createConstrainedRuns(scenarios=[ssp], year_end=year_end)

f_imo_continuous=fair_tools.createConstrainedRuns(scenarios=[ssp], year_end=year_end)
f_imo_1yr=fair_tools.createConstrainedRuns(scenarios=[ssp], year_end=year_end)
f_imo_gettelman=fair_tools.createConstrainedRuns(scenarios=[ssp], year_end=year_end)

imo_forcing=71e-3

imo_forcing_gettelman=0.12

f_imo_continuous.forcing.loc[dict(specie='Volcanic', timebounds=slice(2020,2301))]=f_imo_continuous.forcing.loc[dict(specie='Volcanic', timebounds=slice(2020,2301))]+imo_forcing
f_imo_gettelman.forcing.loc[dict(specie='Volcanic', timebounds=slice(2020,2301))]=f_imo_gettelman.forcing.loc[dict(specie='Volcanic', timebounds=slice(2020,2301))]+imo_forcing_gettelman


f_imo_1yr.forcing.loc[dict(specie='Volcanic', timebounds=slice(2020,2021))]=f_imo_1yr.forcing.loc[dict(specie='Volcanic', timebounds=slice(2020,2021))]+imo_forcing


# Calculate TCRE
tcre, sat_1000gtc = fair_tools.compute_tcre()



# f_imo_continuous.properties['Aerosol-radiation interactions']['input_mode']='forcing'
# f_imo_continuous.properties['Aerosol-cloud interactions']['input_mode']='forcing'

# f_imo_continuous.forcing.loc[dict(specie='Aerosol-radiation interactions')]=5
# f_imo_continuous.forcing.loc[dict(specie='Aerosol-cloud interactions')]=5

f_base.run()
f_imo_continuous.run()
f_imo_gettelman.run()
f_imo_1yr.run()


#Rebase temperature
f_base=fair_tools.rebase_temperature(f_base)
f_imo_continuous=fair_tools.rebase_temperature(f_imo_continuous)
f_imo_gettelman=fair_tools.rebase_temperature(f_imo_gettelman)
f_imo_1yr=fair_tools.rebase_temperature(f_imo_1yr)

# Calculate temperature anomaly with respect to baseline
sat_base=f_base.temperature.sel(layer=0)
sat_continuous=f_imo_continuous.temperature.sel(layer=0)
sat_gettelman=f_imo_gettelman.temperature.sel(layer=0)
sat_1yr=f_imo_1yr.temperature.sel(layer=0)

dsat_continuous=sat_continuous-sat_base
dsat_gettelman=sat_gettelman-sat_base
dsat_1yr=sat_1yr-sat_base

# %%  Calculate GTP of imo regulation
gtp_timescales=[20,50,100]
gtp=pd.DataFrame(index=gtp_timescales, columns=['Continuous','1yr'])

for gtp_timescale in gtp_timescales:
    # Divide temperature response (K) by TCRE (K/GtCO2) to get Gt CO2
    gtp.loc[gtp_timescale,'Continuous']=float((dsat_continuous.sel(timebounds=2020+gtp_timescale)/tcre).mean(dim='config'))
    gtp.loc[gtp_timescale,'1yr']=float((dsat_1yr.sel(timebounds=2020+gtp_timescale)/tcre).mean(dim='config'))




# %%  Figure on sat between scenarios
fig1, ax1 = pl.subplots(1, 2, figsize=(10, 5))

# Left panel: absolute temperatures
sat_base.mean(dim='config').plot(ax=ax1[0], label='Baseline (SSP2-4.5)')
sat_continuous.mean(dim='config').plot(ax=ax1[0], label='IMO 2020 - Continuous')
ax1[0].legend()
ax1[0].set_xlim((2020, year_end))
ax1[0].set_ylim((1, 3))
ax1[0].set_title('Global mean surface temperature\nrelative to 1850-1900')
ax1[0].set_xlabel('Year')
ax1[0].set_ylabel('°C')

# Right panel: temperature difference from baseline
dsat_continuous.mean(dim='config').plot(ax=ax1[1], label='Continuous')
dsat_1yr.mean(dim='config').plot(ax=ax1[1], label='1-year')
ax1[1].set_xlim((2020, year_end))
ax1[1].legend()
ax1[1].set_title('Global mean surface temperature\nrelative to Baseline')
ax1[1].set_xlabel('Year')
ax1[1].set_ylabel('°C')

# === Add secondary y-axis for cumulative emissions ===
# TCRE already in °C/GtCO2 — extract scalar from xarray
tcre_mean = tcre.mean().item()

# Forward: °C → GtCO2, Inverse: GtCO2 → °C
def temp_to_emissions(temp):
    return temp / tcre_mean

def emissions_to_temp(emis):
    return emis * tcre_mean

secax = ax1[1].secondary_yaxis('right', functions=(temp_to_emissions, emissions_to_temp))
secax.set_ylabel('Cumulative emissions difference (GtCO₂)')
# Optional: Set ticks manually, e.g.
# secax.set_yticks(np.arange(0, 2.1, 0.5))

# Save the figure
fig1.savefig(figpath / 'temperature.png', dpi=150)

# %%  Figure on sat between scenarios using Gettelman et al. (2024) forcing of 0.12 Wm-2.
fig2, ax2= pl.subplots(1,2)
(sat_base-sat_base.sel(timebounds=2019)).mean(dim='config').plot(ax=ax2[0], label='Baseline')
(sat_gettelman-sat_gettelman.sel(timebounds=2019)).mean(dim='config').plot(ax=ax2[0], label='IMO 2020 - Continuous')
ax2[0].legend()
ax2[0].set_xlim((2015,2030))
ax2[0].set_ylim((-0.1,0.5))


dsat_gettelman.mean(dim='config').plot(ax=ax2[1], label='Continuous')

ax2[1].set_xlim((2015,2030))
ax2[1].legend()
# ax1[1].set_ylim((0,2.5e-2))



# %%  Figure to demonstrate TCRE
# One point per config at the calibration's 1000 Gt C diagnostic (the
# calibration archive only stores this scalar snapshot per config, not a
# full cumulative-emissions/temperature trajectory for the constrained
# ensemble — see fair_tools.compute_tcre).
fig3, ax3= pl.subplots(1,1)
ax3.scatter(np.full_like(sat_1000gtc, 1000.0), sat_1000gtc, color='tab:grey', s=5, alpha=0.5)
ax3.set_xlim(0,5000)
ax3.set_xlabel('Cumulative CO$_2$ emissions (Gt C)')
ax3.set_ylabel('Temperature change (°C)')
fig3.savefig(figpath / 'cum_emi_temperature.png', dpi=150)

