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
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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

# TCRE already in °C/GtCO2 — extract scalar from xarray. Used as the
# 'shortcut' TCRE value throughout (fig1's secondary axis, and the
# exact-vs-shortcut diagnostics below).
tcre_mean = tcre.mean().item()



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
scenario_labels = ['Continuous', '1yr']

# Per-config CO2-equivalent emissions (GtCO2) at each GTP timescale, keyed by
# (scenario_label, gtp_timescale) -> DataArray with dim 'config'.
# 'exact' divides each member's own temperature response by that member's own
# TCRE. 'shortcut' keeps that same member's temperature response but divides
# by the single ensemble-mean TCRE (tcre_mean) instead -- isolating exactly
# the effect of replacing per-member TCRE with the ensemble mean.
exact_members = {}
shortcut_members = {}

for gtp_timescale in gtp_timescales:
    year = 2020 + gtp_timescale
    for label, dsat in [('Continuous', dsat_continuous), ('1yr', dsat_1yr)]:
        # Divide temperature response (K) by TCRE (K/GtCO2) to get Gt CO2.
        # squeeze 'scenario' (size 1, single SSP) so the stored arrays are
        # 1-D over 'config' only -- needed for hist()/scatter() below, which
        # would otherwise treat each config as its own length-1 series.
        dsat_at_year = dsat.sel(timebounds=year).squeeze('scenario', drop=True)
        exact_members[(label, gtp_timescale)] = dsat_at_year / tcre
        shortcut_members[(label, gtp_timescale)] = dsat_at_year / tcre_mean

# %%  Save GTP tables (mean + 66%/95% range across ensemble members) to CSV,
# one file per method (exact = per-member TCRE, shortcut = ensemble-mean TCRE).
outputpath = Path('output')
outputpath.mkdir(exist_ok=True)

gtp_quantile_levels = [0.025, 0.17, 0.83, 0.975]
gtp_quantile_names = ['p2.5', 'p17', 'p83', 'p97.5']

def build_gtp_table(members_dict):
    table = pd.DataFrame(index=gtp_timescales)
    for label in scenario_labels:
        means = []
        quantiles = []
        for gtp_timescale in gtp_timescales:
            vals = members_dict[(label, gtp_timescale)]
            means.append(float(vals.mean(dim='config')))
            quantiles.append(vals.quantile(gtp_quantile_levels, dim='config').values)
        table[f'{label}_mean'] = means
        quantiles = np.array(quantiles)
        for i, name in enumerate(gtp_quantile_names):
            table[f'{label}_{name}'] = quantiles[:, i]
    return table

gtp_exact = build_gtp_table(exact_members)
gtp_shortcut = build_gtp_table(shortcut_members)

gtp_exact.to_csv(outputpath / 'gtp_exact.csv')
gtp_shortcut.to_csv(outputpath / 'gtp_shortcut.csv')




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

# Order-of-magnitude sanity check at one timestep, including per-member spread
example_year = 2070
exact_val = float(co2_equiv_continuous_exact.sel(timebounds=example_year))
shortcut_val = float(co2_equiv_continuous_shortcut.sel(timebounds=example_year))
exact_std = float(exact_members[('Continuous', 50)].std(dim='config'))
shortcut_std = float(shortcut_members[('Continuous', 50)].std(dim='config'))
print(f'At {example_year}, Continuous scenario: exact={exact_val:.4g} GtCO2, '
      f'shortcut={shortcut_val:.4g} GtCO2, '
      f'diff={100*(shortcut_val-exact_val)/exact_val:.2f}%, '
      f'per-member std: exact={exact_std:.4g}, shortcut={shortcut_std:.4g} GtCO2')

# %%  Quantile uncertainty data for fig1b/fig1c below.
# 66% range = 17th-83rd percentile, 95% range = 2.5th-97.5th percentile.
# Mean lines reuse co2_equiv_*_exact / dsat_*.mean() computed above rather
# than adding a 0.5 quantile, since mean (not median) is the requested
# central estimate.
quantile_levels = [0.025, 0.17, 0.83, 0.975]
scenario_dsat = {'Continuous': dsat_continuous, '1yr': dsat_1yr}

dsat_quantiles = {
    label: da.quantile(quantile_levels, dim='config').squeeze('scenario', drop=True)
    for label, da in scenario_dsat.items()
}

# Per-member exact: divide each member's own dT by that member's own TCRE
# *before* taking quantiles, so the spread reflects the real TCRE/warming
# correlation across members -- same approach as co2_equiv_*_exact above,
# extended from a single mean value to a full quantile range.
co2_equiv_exact_quantiles = {
    label: (da / tcre).quantile(quantile_levels, dim='config').squeeze('scenario', drop=True)
    for label, da in scenario_dsat.items()
}

# Order-of-magnitude sanity check at one timestep.
dt_mean_check = float(dsat_continuous.mean(dim='config').sel(timebounds=example_year))
dt_lo95_check = float(dsat_quantiles['Continuous'].sel(timebounds=example_year, quantile=0.025))
dt_hi95_check = float(dsat_quantiles['Continuous'].sel(timebounds=example_year, quantile=0.975))
print(f'At {example_year}, Continuous dT: mean={dt_mean_check:.4g} degC, '
      f'95% range=[{dt_lo95_check:.4g}, {dt_hi95_check:.4g}] degC')

co2_lo95_check = float(co2_equiv_exact_quantiles['Continuous'].sel(timebounds=example_year, quantile=0.025))
co2_hi95_check = float(co2_equiv_exact_quantiles['Continuous'].sel(timebounds=example_year, quantile=0.975))
print(f'At {example_year}, Continuous CO2-equiv (exact, per-member TCRE): '
      f'mean={exact_val:.4g} GtCO2, 95% range=[{co2_lo95_check:.4g}, {co2_hi95_check:.4g}] GtCO2')

unc_colors = {'Continuous': 'tab:blue', '1yr': 'tab:orange'}

class HandlerUncertaintyBand(HandlerBase):
    '''Draws a 95%-band rectangle (full height), a 66%-band rectangle nested
    inside it (so the wider 95% shading stays visible as a margin, matching
    the plot itself), and the mean line through the middle -- all as one
    legend swatch.'''
    def __init__(self, color, inner_frac=0.55, **kwargs):
        self.color = color
        self.inner_frac = inner_frac
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        outer = Rectangle((-xdescent, -ydescent), width, height,
                           facecolor=self.color, alpha=0.15, transform=trans)
        inner_height = height * self.inner_frac
        inner = Rectangle((-xdescent, -ydescent + (height - inner_height) / 2), width, inner_height,
                           facecolor=self.color, alpha=0.3, transform=trans)
        line = Line2D([-xdescent, -xdescent + width], [-ydescent + height / 2] * 2,
                      color=self.color, transform=trans)
        return [outer, inner, line]

def combined_uncertainty_legend(ax, labels):
    '''One legend entry per scenario: 95%-band + 66%-band + mean line in a single swatch.'''
    handles = [Line2D([], [], color=unc_colors[label]) for label in labels]
    handler_map = {handle: HandlerUncertaintyBand(unc_colors[label])
                    for handle, label in zip(handles, labels)}
    ax.legend(handles, labels, handler_map=handler_map, handlelength=2.5, handleheight=1.5)

# %%  Figure 1b: dT with 66%/95% quantile bands + secondary axis (shortcut TCRE)
fig1b, ax1b = pl.subplots(1, 1, figsize=(6, 5))

for label, dsat_mean_series in [('Continuous', dsat_continuous.mean(dim='config')),
                                 ('1yr', dsat_1yr.mean(dim='config'))]:
    q = dsat_quantiles[label]
    color = unc_colors[label]
    ax1b.fill_between(q.timebounds, q.sel(quantile=0.025), q.sel(quantile=0.975),
                       color=color, alpha=0.15)
    ax1b.fill_between(q.timebounds, q.sel(quantile=0.17), q.sel(quantile=0.83),
                       color=color, alpha=0.3)
    dsat_mean_series.plot(ax=ax1b, color=color)

ax1b.set_xlim((2020, year_end))
combined_uncertainty_legend(ax1b, ['Continuous', '1yr'])
ax1b.set_title('Global mean surface temperature relative to Baseline\n(66%/95% range across ensemble members)')
ax1b.set_xlabel('Year')
ax1b.set_ylabel('°C')

secax1b = ax1b.secondary_yaxis('right', functions=(temp_to_emissions, emissions_to_temp))
secax1b.set_ylabel('Cumulative CO$_2$-warming-equivalent emissions (GtCO₂)\n(dT / mean(TCRE))')

fig1b.savefig(figpath / 'temperature_uncertainty.png', dpi=150)

# %%  Figure 1c: cumulative CO2-warming-equivalent emissions from per-member
# TCRE (exact method), with 66%/95% quantile bands. No secondary axis here:
# unlike fig1b's secondary axis (a single fixed dT->GtCO2 scale factor via
# tcre_mean), the exact per-member relationship maps the same dT to a
# different GtCO2 value depending on which member it came from, which isn't
# representable as a static axis transform -- hence its own panel.
fig1c, ax1c = pl.subplots(1, 1, figsize=(6, 5))

for label, co2_mean_series in [('Continuous', co2_equiv_continuous_exact),
                                ('1yr', co2_equiv_1yr_exact)]:
    q = co2_equiv_exact_quantiles[label]
    color = unc_colors[label]
    ax1c.fill_between(q.timebounds, q.sel(quantile=0.025), q.sel(quantile=0.975),
                       color=color, alpha=0.15)
    ax1c.fill_between(q.timebounds, q.sel(quantile=0.17), q.sel(quantile=0.83),
                       color=color, alpha=0.3)
    co2_mean_series.plot(ax=ax1c, color=color)

ax1c.set_xlim((2020, year_end))
combined_uncertainty_legend(ax1c, ['Continuous', '1yr'])
ax1c.set_title('Cumulative CO$_2$-warming-equivalent emissions of IMO regulation\n(per-member TCRE, 66%/95% range across ensemble members)')
ax1c.set_xlabel('Year')
ax1c.set_ylabel('Cumulative CO$_2$-warming-equivalent emissions (GtCO$_2$)')

fig1c.savefig(figpath / 'co2_equivalent_uncertainty.png', dpi=150)

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

# %%  Per-member distributions: exact (own TCRE) vs shortcut (mean TCRE)
# Small multiples: rows = scenario, columns = GTP timescale. Each panel
# overlays the exact and shortcut per-config distributions of CO2-equivalent
# emissions, so the spread/location of the simplification's error is visible
# directly, rather than just the difference of two already-averaged numbers.
fig5, ax5 = pl.subplots(len(scenario_labels), len(gtp_timescales), figsize=(12, 7), sharex='col')

for row, label in enumerate(scenario_labels):
    for col, gtp_timescale in enumerate(gtp_timescales):
        ax = ax5[row, col]
        exact_vals = exact_members[(label, gtp_timescale)].values
        shortcut_vals = shortcut_members[(label, gtp_timescale)].values
        ax.hist(exact_vals, bins=40, alpha=0.5, label='exact')
        ax.hist(shortcut_vals, bins=40, alpha=0.5, label='shortcut')
        ax.axvline(exact_vals.mean(), color='tab:blue', linestyle='--', lw=1)
        ax.axvline(shortcut_vals.mean(), color='tab:orange', linestyle='--', lw=1)
        ax.set_title(f'{label}, {gtp_timescale} yr')
        if row == len(scenario_labels) - 1:
            ax.set_xlabel('GtCO$_2$')
        if col == 0:
            ax.set_ylabel('Count')

ax5[0, 0].legend()
fig5.suptitle('Per-member CO$_2$-equivalent emissions:\nexact (own TCRE) vs shortcut (mean TCRE)')
fig5.tight_layout()
fig5.savefig(figpath / 'gtp_exact_vs_shortcut_hist.png', dpi=150)

# %%  Per-member paired scatter: exact vs shortcut (same config, same panel layout)
fig6, ax6 = pl.subplots(len(scenario_labels), len(gtp_timescales), figsize=(12, 7))

for row, label in enumerate(scenario_labels):
    for col, gtp_timescale in enumerate(gtp_timescales):
        ax = ax6[row, col]
        exact_vals = exact_members[(label, gtp_timescale)].values
        shortcut_vals = shortcut_members[(label, gtp_timescale)].values
        ax.scatter(shortcut_vals, exact_vals, s=5, alpha=0.4)

        lo = min(shortcut_vals.min(), exact_vals.min())
        hi = max(shortcut_vals.max(), exact_vals.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlim((lo, hi))
        ax.set_ylim((lo, hi))
        ax.set_aspect('equal')
        ax.set_title(f'{label}, {gtp_timescale} yr')
        if row == len(scenario_labels) - 1:
            ax.set_xlabel('Shortcut (GtCO$_2$)')
        if col == 0:
            ax.set_ylabel('Exact (GtCO$_2$)')

fig6.suptitle('Per-member CO$_2$-equivalent emissions:\nshortcut vs exact, paired by ensemble member')
fig6.tight_layout()
fig6.savefig(figpath / 'gtp_exact_vs_shortcut_scatter.png', dpi=150)

