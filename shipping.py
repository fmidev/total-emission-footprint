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

# %%  Exact (per-ensemble-member) vs shortcut CO2-equivalent emissions
# Shortcut (used for fig1's secondary axis above): divide the ensemble-MEAN
# temperature response by the ensemble-MEAN TCRE (a single scalar). This
# implicitly assumes TCRE and the temperature response are uncorrelated
# across configs, which need not hold since both are driven by the same
# per-config climate sensitivity (high-TCRE configs also warm faster).
# Exact: divide each config's own temperature response by that config's own
# TCRE first, then average over configs -- the same approach already used
# for the gtp table above, here extended to the full time series.
co2_equiv_continuous_shortcut = dsat_continuous.mean(dim='config') / tcre_mean
co2_equiv_continuous_exact = (dsat_continuous / tcre).mean(dim='config')

co2_equiv_1yr_shortcut = dsat_1yr.mean(dim='config') / tcre_mean
co2_equiv_1yr_exact = (dsat_1yr / tcre).mean(dim='config')

# Order-of-magnitude sanity check at one timestep
example_year = 2070
exact_val = float(co2_equiv_continuous_exact.sel(timebounds=example_year))
shortcut_val = float(co2_equiv_continuous_shortcut.sel(timebounds=example_year))
print(f'At {example_year}, Continuous scenario: exact={exact_val:.4g} GtCO2, '
      f'shortcut={shortcut_val:.4g} GtCO2, '
      f'diff={100*(shortcut_val-exact_val)/exact_val:.2f}%')

fig4, ax4 = pl.subplots(1, 1, figsize=(6, 5))
co2_equiv_continuous_exact.plot(ax=ax4, label='Continuous (exact)')
co2_equiv_continuous_shortcut.plot(ax=ax4, label='Continuous (shortcut)', linestyle='--')
co2_equiv_1yr_exact.plot(ax=ax4, label='1-year (exact)')
co2_equiv_1yr_shortcut.plot(ax=ax4, label='1-year (shortcut)', linestyle='--')
ax4.set_xlim((2020, year_end))
ax4.legend()
ax4.set_title('CO$_2$-equivalent emissions of IMO regulation\nexact (per-member TCRE) vs shortcut (mean TCRE)')
ax4.set_xlabel('Year')
ax4.set_ylabel('Cumulative CO$_2$-equivalent emissions (GtCO$_2$)')
fig4.savefig(figpath / 'gtp_exact_vs_shortcut_timeseries.png', dpi=150)

# %%  Scatter: shortcut vs exact CO2-equivalent emissions, one point per year
fig5, ax5 = pl.subplots(1, 1, figsize=(5, 5))
sel_years = slice(2020, year_end)
ax5.scatter(co2_equiv_continuous_shortcut.sel(timebounds=sel_years),
            co2_equiv_continuous_exact.sel(timebounds=sel_years),
            s=8, alpha=0.5, label='Continuous')
ax5.scatter(co2_equiv_1yr_shortcut.sel(timebounds=sel_years),
            co2_equiv_1yr_exact.sel(timebounds=sel_years),
            s=8, alpha=0.5, label='1-year')

lo = min(ax5.get_xlim()[0], ax5.get_ylim()[0])
hi = max(ax5.get_xlim()[1], ax5.get_ylim()[1])
ax5.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1')
ax5.set_xlim((lo, hi))
ax5.set_ylim((lo, hi))
ax5.set_aspect('equal')
ax5.set_xlabel('Shortcut: mean(ΔT) / mean(TCRE)  (GtCO$_2$)')
ax5.set_ylabel('Exact: mean(ΔT / TCRE)  (GtCO$_2$)')
ax5.set_title('CO$_2$-equivalent emissions:\nshortcut vs exact per-member calculation')
ax5.legend()
fig5.savefig(figpath / 'gtp_shortcut_vs_exact_scatter.png', dpi=150)

