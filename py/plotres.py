#!/usr/local/anaconda3/envs/py36 python
# -*- coding: utf-8 -*-

# Plotting
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as pl
import seaborn as sns

sns.set_style("ticks")

# Imports
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import interpolate
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve
from scipy.signal import medfilt
import astropy.units as u
from astropy.time import Time
from util import *


import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

params = {
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.usetex": True,
    "figure.figsize": [8, 8 / 1.61],
}
mpl.rcParams.update(params)


def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f


def main():
    root_dir = "/Users/jonatanselsing/Work/work_rawDATA/Crab_Pulsar/final/"
    bin_f = 10

    # OBs = ["OB1", "OB2", "OB3", "OB4", "OB5", "OB6", "OB7", "OB8", "OB9", "OB10", "OB11", "OB12", "OB13", "OB14", "OB15", "OB16", "OB17", "OB18"]

    OBs = ["OB9", "OB9_pip"]
    z = 0
    colors = sns.color_palette("viridis", len(OBs))
    wl_out, flux_out, error_out = 0, 0, 0
    for ii, kk in enumerate(OBs):
        off = 0  # (len(OBs) - ii) * 2e-17
        mult = 1.0

        ############################## OB ##############################
        f = fits.open(root_dir + "UVB%s.fits" % kk)
        wl = 10.0 * f[1].data.field("WAVE").flatten()

        q = f[1].data.field("QUAL").flatten()
        try:
            t = f[1].data.field("TRANS").flatten()
        except:
            t = np.ones_like(q)
        mask_wl = (wl > 3200) & (wl < 5550)
        mask_qual = ~q.astype("bool")
        flux = interpolate.interp1d(
            wl[mask_qual],
            f[1].data.field("FLUX").flatten()[mask_qual],
            bounds_error=False,
            fill_value=0,
        )

        error = interpolate.interp1d(
            wl[mask_qual],
            f[1].data.field("ERR").flatten()[mask_qual],
            bounds_error=False,
            fill_value=0,
        )
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot)
        error = error(wl_plot)
        pl.plot(
            wl_plot[::1] / (1 + z),
            off + mult * flux[::1],
            lw=0.3,
            color="black",
            alpha=0.2,
            rasterized=True,
        )

        wl_stitch, flux_stitch, error_stitch = wl_plot, flux, error
        b_wl, b_f, b_e, b_q = bin_spectrum(
            wl_plot,
            flux,
            error,
            np.zeros_like(flux).astype("bool"),
            round_up_to_odd(bin_f),
        )
        wl_out, flux_out, error_out = b_wl, b_f, b_e
        pl.plot(
            b_wl / (1 + z),
            off + mult * b_f,
            color=colors[ii],
            linestyle="steps-mid",
            rasterized=True,
        )

        max_v, min_v = max(medfilt(flux, 101)), min(medfilt(flux, 101))

        f = fits.open(root_dir + "VIS%s.fits" % kk)
        wl = 10.0 * f[1].data.field("WAVE").flatten()
        q = f[1].data.field("QUAL").flatten()
        try:
            t = f[1].data.field("TRANS").flatten()
        except:
            t = np.ones_like(wl)

        mask_wl = (wl > 5650) & (wl < 10000)
        mask_qual = ~q.astype("bool")

        flux = interpolate.interp1d(
            wl[mask_qual],
            f[1].data.field("FLUX").flatten()[mask_qual],
            bounds_error=False,
            fill_value=0,
        )
        error = interpolate.interp1d(
            wl[mask_qual],
            f[1].data.field("ERR").flatten()[mask_qual],
            bounds_error=False,
            fill_value=0,
        )
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot) / t[mask_wl]
        error = error(wl_plot) / t[mask_wl]

        pl.plot(
            wl_plot[::1] / (1 + z),
            off + mult * flux[::1],
            lw=0.3,
            color="black",
            alpha=0.2,
            rasterized=True,
        )
        wl_stitch, flux_stitch, error_stitch = (
            np.concatenate([wl_stitch, wl_plot]),
            np.concatenate([flux_stitch, flux]),
            np.concatenate([error_stitch, error]),
        )
        b_wl, b_f, b_e, b_q = bin_spectrum(
            wl_plot,
            flux,
            error,
            np.zeros_like(flux).astype("bool"),
            round_up_to_odd(bin_f),
        )
        wl_out, flux_out, error_out = b_wl, b_f, b_e
        pl.plot(
            b_wl / (1 + z),
            off + mult * b_f,
            color=colors[ii],
            linestyle="steps-mid",
            rasterized=True,
        )

        max_v, min_v = max(max(medfilt(flux, 101)), max_v), min(
            min(medfilt(flux, 101)), min_v
        )

        f = fits.open(root_dir + "NIR%s.fits" % kk)
        wl = 10.0 * f[1].data.field("WAVE").flatten()
        q = f[1].data.field("QUAL").flatten()

        try:
            t = f[1].data.field("TRANS").flatten()
        except:
            t = np.ones_like(wl)

        mask_wl = (wl > 10500) & (wl < 25000) & (t > 0.3)
        mask_qual = ~q.astype("bool")
        flux = interpolate.interp1d(
            wl[mask_qual],
            f[1].data.field("FLUX").flatten()[mask_qual],
            bounds_error=False,
            fill_value=0,
        )
        error = interpolate.interp1d(
            wl[mask_qual],
            f[1].data.field("ERR").flatten()[mask_qual],
            bounds_error=False,
            fill_value=0,
        )
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot) / t[mask_wl]
        error = error(wl_plot) / t[mask_wl]

        pl.plot(
            wl_plot[::1] / (1 + z),
            off + mult * flux[::1],
            lw=0.3,
            color="black",
            alpha=0.2,
            rasterized=True,
        )

        wl_stitch, flux_stitch, error_stitch = (
            np.concatenate([wl_stitch, wl_plot]),
            np.concatenate([flux_stitch, flux]),
            np.concatenate([error_stitch, error]),
        )
        b_wl, b_f, b_e, b_q = bin_spectrum(
            wl_plot,
            flux,
            error,
            np.zeros_like(flux).astype("bool"),
            round_up_to_odd(bin_f / 3),
        )
        wl_out, flux_out, error_out = b_wl, b_f, b_e
        pl.plot(
            b_wl / (1 + z),
            off + mult * b_f,
            color=colors[ii],
            linestyle="steps-mid",
            rasterized=True,
            label="XSH: %s" % str(f[0].header["DATE-OBS"]),
        )
        max_v, min_v = max(max(medfilt(flux, 101)), max_v), min(
            min(medfilt(flux, 101)), min_v
        )

        pl.axvspan(5550, 5650, color="grey", alpha=0.2)
        pl.axvspan(10000, 10500, color="grey", alpha=0.2)
        pl.axvspan(12600, 12800, color="grey", alpha=0.2)
        pl.axvspan(13500, 14500, color="grey", alpha=0.2)
        pl.axvspan(18000, 19500, color="grey", alpha=0.2)

        # pl.ylim(-1e-18, 1.2 * max_v)
        # pl.xlim(2500, 20000)
        pl.xlabel(r"Observed wavelength  [$\mathrm{\AA}$]")
        pl.ylabel(
            r"Flux density [$\mathrm{erg} \mathrm{s}^{-1} \mathrm{cm}^{-1} \mathrm{\AA}^{-1}$]"
        )
        pl.axhline(0, linestyle="dashed", color="black")
        pl.legend()
        pl.savefig(root_dir + "%s.pdf" % kk)
        pl.show()
        pl.clf()

        wl = np.arange(min(wl_stitch), max(wl_stitch), np.median(np.diff(wl_stitch)))
        f = interpolate.interp1d(wl_stitch, flux_stitch)
        g = interpolate.interp1d(wl_stitch, error_stitch)
        np.savetxt(
            root_dir + "%s_stitched.dat" % kk,
            list(zip(wl, f(wl), g(wl))),
            fmt="%1.2f %1.4e %1.4e",
        )
        # pl.show()


if __name__ == "__main__":
    main()
