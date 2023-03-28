#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from scipy import signal
from scipy import interpolate
from scipy.special import wofz, erf
from astropy.modeling import models
from astropy.io import fits
from astropy import wcs


__all__ = [
    "correct_for_dust",
    "bin_image",
    "avg",
    "gaussian",
    "voigt",
    "two_voigt",
    "slit_loss",
    "convert_air_to_vacuum",
    "convert_vacuum_to_air",
    "inpaint_nans",
    "bin_spectrum",
    "form_nodding_pairs",
    "find_nearest",
    "get_slitloss",
    "Moffat1D",
    "Two_Moffat1D",
    "filter_bad_values",
    "XshOrder2D",
    "ADC_corr_guess",
]


def get_slitloss(seeing_fwhm, slit_width):
    # Generate image parameters
    img_size = 100

    arcsec_to_pix = img_size / (seeing_fwhm * 3)  # Assumes a 3xfwhm arcsec image
    slit_width_pix = arcsec_to_pix * slit_width
    seeing_pix = arcsec_to_pix * seeing_fwhm

    x, y = np.mgrid[:img_size, :img_size]
    source_pos = [int(img_size / 2), int(img_size / 2)]

    # Simulate source Moffat
    beta = 4.765
    gamma = seeing_pix / (2 * np.sqrt(2 ** (1 / beta) - 1))
    source = models.Moffat2D.evaluate(
        x, y, 1, source_pos[0], source_pos[1], gamma, beta
    )

    # Define slit mask
    mask = slice(
        int(source_pos[1] - slit_width_pix / 2), int(source_pos[1] + slit_width_pix / 2)
    )
    sl = np.trapz(np.trapz(source)) / np.trapz(np.trapz(source[:, mask]))

    return sl


def Moffat1D(x, amplitude, x_0, fwhm, c=0, a=0):
    beta = 4.765
    gamma = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    return models.Moffat1D.evaluate(x, amplitude, x_0, gamma, 4.765) + c + a * x


def Two_Moffat1D(
    x, amplitude_1=1, x_0_1=0, fwhm_1=1, c=0, a=0, amplitude_2=1, x_0_2=0, fwhm_2=1
):
    beta = 4.765
    gamma_1 = fwhm_1 / (2 * np.sqrt(2 ** (1 / beta) - 1))
    gamma_2 = fwhm_2 / (2 * np.sqrt(2 ** (1 / beta) - 1))
    return (
        models.Moffat1D.evaluate(x, amplitude_1, x_0_1, gamma_1, 4.765)
        + c
        + a * x
        + models.Moffat1D.evaluate(x, amplitude_2, x_0_2, gamma_2, 4.765)
    )


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def gaussian(x, amp, cen, sigma):
    # Simple Gaussian
    return amp * np.exp(-((x - cen) ** 2) / sigma**2)


def voigt(x, amp=1, cen=0, sigma=1, gamma=0, c=0, a=0):
    """1 dimensional voigt function.
    see http://en.wikipedia.org/wiki/Voigt_profile
    """
    # Penalize negative values
    if sigma <= 0:
        amp = 1e10
    if gamma <= 0:
        amp = 1e10
    if amp <= 0:
        amp = 1e10
    z = (x - cen + 1j * gamma) / (sigma * np.sqrt(2.0))

    return amp * wofz(z).real / (sigma * np.sqrt(2 * np.pi)) + c + a * x


def two_voigt(
    x, amp=1, cen=0, sigma=1, gamma=0, c=0, a=0, amp2=0.0, cen2=-1, sig2=0.5, gam2=0
):
    """1 dimensional voigt function.
    see http://en.wikipedia.org/wiki/Voigt_profile
    """
    # Penalize negative values
    if sigma <= 0:
        amp = 1e10
    if sig2 <= 0:
        amp = 1e10
    if gamma <= 0:
        amp = 1e10
    if gam2 <= 0:
        amp = 1e10
    if amp <= 0:
        amp = 1e10
    if amp2 <= 0:
        amp2 = 1e10
    z = (x - cen + 1j * gamma) / (sigma * np.sqrt(2.0))
    z2 = (x - cen2 + 1j * gam2) / (sig2 * np.sqrt(2.0))
    return (
        amp * wofz(z).real / (sigma * np.sqrt(2 * np.pi))
        + c
        + a * x
        + amp2 * wofz(z2).real / (sig2 * np.sqrt(2 * np.pi))
    )


def slit_loss(g_sigma, slit_width, l_sigma=False):
    """
    Calculates the slit-loss based on the seeing sigma and slit width in arcsec
    """
    # With pure Gaussian, do the analytical solution
    try:
        if not l_sigma:
            # FWHM = 2 * np.sqrt(2 * np.log(2))
            return 1 / erf((slit_width / 2) / (np.sqrt(2) * (g_sigma)))
    except:
        pass
    # For the voigt, calculate the integral numerically.
    x = np.arange(-10, 10, 0.01)
    v = [voigt(x, sigma=kk, gamma=l_sigma[ii]) for ii, kk in enumerate(g_sigma)]

    mask = (x > -slit_width / 2) & (x < slit_width / 2)
    sl = np.zeros_like(g_sigma)
    for ii, kk in enumerate(g_sigma):
        sl[ii] = np.trapz(v[ii], x) / np.trapz(v[ii][mask], x[mask])

    # pl.plot(x, v[0])
    # print(slit_width, g_sigma, l_sigma, sl)
    # pl.show()
    # exit()

    return sl


def avg(flux, error, mask=None, axis=2, weight=False, weight_map=None):
    """Calculate the weighted average with errors
    ----------
    flux : array-like
        Values to take average of
    error : array-like
        Errors associated with values, assumed to be standard deviations.
    mask : array-like
        Array of bools, where true means a masked value.
    axis : int, default 0
        axis argument passed to numpy

    Returns
    -------
    average, error : tuple

    Notes
    -----
    """
    try:
        if not mask:
            mask = np.zeros_like(flux).astype("bool")
    except:
        pass
        # print("All values are masked... Returning nan")
        # if np.sum(mask.astype("int")) == 0:
        #     return np.nan, np.nan, np.nan

    # Normalize to avoid numerical issues in flux-calibrated data
    norm = abs(np.ma.median(flux[flux > 0]))
    if norm == np.nan or norm == np.inf or norm == 0:
        print(
            "Nomalization factor in avg has got a bad value. It's "
            + str(norm)
            + " ... Replacing with 1"
        )

    flux_func = flux.copy() / norm
    error_func = error.copy() / norm

    # Calculate average based on supplied weight map
    if weight_map is not None:
        # Remove non-contributing pixels
        flux_func[mask] = 0
        error_func[mask] = 0
        # https://physics.stackexchange.com/questions/15197/how-do-you-find-the-uncertainty-of-a-weighted-average?newreg=4e2b8a1d87f04c01a82940d234a07fc5
        average = np.ma.sum(flux_func * weight_map, axis=axis) / np.ma.sum(
            weight_map, axis=axis
        )
        variance = (
            np.ma.sum(error_func**2 * weight_map**2, axis=axis)
            / np.ma.sum(weight_map, axis=axis) ** 2
        )

    # Inverse variance weighted average
    elif weight:
        ma_flux_func = np.ma.array(flux_func, mask=mask)
        ma_error_func = np.ma.array(error_func, mask=mask)
        w = 1.0 / (ma_error_func**2.0)
        average = np.ma.sum(ma_flux_func * w, axis=axis) / np.ma.sum(w, axis=axis)
        variance = 1.0 / np.ma.sum(w, axis=axis)
        if not isinstance(average, float):
            # average[average.mask] = np.nan
            average = average.data
            # variance[variance.mask] = np.nan
            variance = variance.data

    # Normal average
    elif not weight:
        # Number of pixels in the mean
        n = np.ma.sum(np.array(~mask).astype("int"), axis=axis)
        # Remove non-contributing pixels
        flux_func[mask] = 0
        error_func[mask] = 0
        # mean
        average = (1 / n) * np.ma.sum(flux_func, axis=axis)
        # probagate errors
        variance = (1 / n**2) * np.ma.sum(error_func**2.0, axis=axis)

    mask = (np.ma.sum((~mask).astype("int"), axis=axis) == 0).astype("int")
    return (average * norm, np.sqrt(variance) * norm, mask)


def correct_for_dust(wavelength, ra, dec):
    """Query IRSA dust map for E(B-V) value and returns reddening array
    ----------
    wavelength : numpy array-like
        Wavelength values for which to return reddening
    ra : float
        Right Ascencion in degrees
    dec : float
        Declination in degrees

    Returns
    -------
    reddening : numpy array

    Notes
    -----
    For info on the dust maps, see http://irsa.ipac.caltech.edu/applications/DUST/
    """

    from astroquery.irsa_dust import IrsaDust
    import astropy.coordinates as coord
    import astropy.units as u

    C = coord.SkyCoord(ra * u.deg, dec * u.deg, frame="fk5")
    # dust_image = IrsaDust.get_images(C, radius=2 *u.deg, image_type='ebv', timeout=60)[0]
    # ebv = np.mean(dust_image[0].data[40:42, 40:42])
    dust_table = IrsaDust.get_query_table(C, section="ebv", timeout=60)
    ebv = dust_table["ext SandF ref"][0]

    from dust_extinction.parameter_averages import F04

    # initialize the model
    ext = F04(Rv=3.1)
    reddening = 1 / ext.extinguish(wavelength * u.angstrom, Ebv=ebv)
    return reddening, ebv


def bin_spectrum(wl, flux, error, mask, binh, weight=False):
    """Bin low S/N 1D data from xshooter
    ----------
    flux : np.array containing 2D-image flux
        Flux in input image
    error : np.array containing 2D-image error
        Error in input image
    binh : int
        binning along x-axis

    Returns
    -------
    binned fits image
    """

    print("Binning image by a factor: " + str(binh))
    if binh == 1:
        return wl, flux, error, mask

    # Outsize
    size = flux.shape[0]
    outsize = int(np.round(size / binh))

    # Containers
    wl_out = np.zeros((outsize))
    res = np.zeros((outsize))
    reserr = np.zeros((outsize))
    resbp = np.zeros((outsize))

    for ii in np.arange(0, size - binh, binh):
        # Find psotions in new array
        slice(ii, ii + binh)
        h_index = int((ii + binh) / binh) - 1
        # Construct weighted average and weighted std along binning axis
        res[h_index], reserr[h_index], resbp[h_index] = avg(
            flux[ii : ii + binh],
            error[ii : ii + binh],
            mask=mask[ii : ii + binh],
            axis=0,
            weight=weight,
        )
        wl_out[h_index] = np.ma.median(wl[ii : ii + binh], axis=0)

    return wl_out[1:-1], res[1:-1], reserr[1:-1], resbp[1:-1]


def bin_image(flux, error, mask, binh, weight=False):
    """Bin low S/N 2D data from xshooter
    ----------
    flux : np.array containing 2D-image flux
        Flux in input image
    error : np.array containing 2D-image error
        Error in input image
    binh : int
        binning along x-axis

    Returns
    -------
    binned fits image
    """

    # print("Binning image by a factor: "+str(binh))
    if binh == 1:
        return flux, error

    # Outsize
    v_size, h_size = flux.shape
    outsizeh = int(h_size / binh)

    # Containers
    res = np.zeros((v_size, outsizeh))
    reserr = np.zeros((v_size, outsizeh))
    resbpmap = np.zeros((v_size, outsizeh))

    flux_tmp = flux.copy()
    for ii in np.arange(0, h_size - binh, binh):
        # Find psotions in new array
        slice(ii, ii + binh)
        h_index = int((ii + binh) / binh - 1)

        # Sigma clip before binning to remove noisy pixels with bad error estimate.
        # clip_mask = sigma_clip(flux[:, ii:ii + binh], axis=1)

        # Combine masks
        mask_comb = mask[:, ii : ii + binh].astype("bool")  # | clip_mask.mask

        # Construct weighted average and weighted std along binning axis
        res[:, h_index], reserr[:, h_index], resbpmap[:, h_index] = avg(
            flux_tmp[:, ii : ii + binh],
            error[:, ii : ii + binh],
            mask=mask_comb,
            axis=1,
            weight=weight,
        )

    return res, reserr, resbpmap


def convert_air_to_vacuum(air_wave):
    # convert air to vacuum. Based onhttp://idlastro.gsfc.nasa.gov/ftp/pro/astro/airtovac.pro
    # taken from https://github.com/desihub/specex/blob/master/python/specex_air_to_vacuum.py

    sigma2 = (1e4 / air_wave) ** 2
    fact = 1.0 + 5.792105e-2 / (238.0185 - sigma2) + 1.67917e-3 / (57.362 - sigma2)
    vacuum_wave = air_wave * fact

    # comparison with http://www.sdss.org/dr7/products/spectra/vacwavelength.html
    # where : AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)
    # air_wave=numpy.array([4861.363,4958.911,5006.843,6548.05,6562.801,6583.45,6716.44,6730.82])
    # expected_vacuum_wave=numpy.array([4862.721,4960.295,5008.239,6549.86,6564.614,6585.27,6718.29,6732.68])
    return vacuum_wave


def convert_vacuum_to_air(vac_wave):
    # convert vacuum to air
    # taken from http://www.sdss.org/dr7/products/spectra/vacwavelength.html

    air_wave = vac_wave / (
        1.0 + 2.735182e-4 + 131.4182 / vac_wave**2 + 2.76249e8 / vac_wave**4
    )
    return air_wave


def inpaint_nans(im, kernel_size=5):
    # Taken from http://stackoverflow.com/a/21859317/6519723
    ipn_kernel = np.ones((kernel_size, kernel_size))  # kernel for inpaint_nans
    ipn_kernel[int(kernel_size / 2), int(kernel_size / 2)] = 0

    nans = np.isnan(im)
    while np.sum(nans) > 0:
        im[nans] = 0
        vNeighbors = signal.convolve2d(
            (nans == False), ipn_kernel, mode="same", boundary="symm"
        )
        im2 = signal.convolve2d(im, ipn_kernel, mode="same", boundary="symm")
        im2[vNeighbors > 0] = im2[vNeighbors > 0] / vNeighbors[vNeighbors > 0]
        im2[vNeighbors == 0] = np.nan
        im2[(nans == False)] = im[(nans == False)]
        im = im2
        nans = np.isnan(im)
    return im


def form_nodding_pairs(flux_cube, error_cube, bpmap_cube, naxis2, pix_offsety):
    if not len(pix_offsety) % 2 == 0:
        print("")
        print("Attempting to form nodding pairs out of an uneven number of images ...")
        print("Discarding last image ...")
        print("")
        pix_offsety = pix_offsety[:-1]

    flux_cube_out = np.zeros(flux_cube.shape)
    error_cube_out = np.zeros(error_cube.shape)
    bpmap_cube_out = np.ones(bpmap_cube.shape) * 10

    # Make mask based on the bad-pixel map, the edge mask and the sigma-clipped mask
    mask_cube = bpmap_cube != 0

    # Setting masks
    flux_cube[mask_cube] = 0
    error_cube[mask_cube] = 0
    bpmap_cube[mask_cube] = 1

    # Finding the indices of the container in which to put image.
    offv = np.zeros_like(pix_offsety)
    for ii, kk in enumerate(pix_offsety):
        offv[ii] = kk - min(pix_offsety)

    # Define slices where to put image
    v_range = []
    for ii, kk in enumerate(offv):
        v_range.append(slice(kk, naxis2 + kk))

    # From A-B and B-A pairs
    alter = 1
    for ii, kk in enumerate(v_range):
        flux_cube_out[kk, :, ii] = (
            flux_cube[kk, :, ii] - flux_cube[v_range[ii + alter], :, ii + alter]
        )
        error_cube_out[kk, :, ii] = np.sqrt(
            error_cube[kk, :, ii] ** 2.0
            + error_cube[v_range[ii + alter], :, ii + alter] ** 2.0
        )
        bpmap_cube_out[kk, :, ii] = (
            bpmap_cube[kk, :, ii] + bpmap_cube[v_range[ii + alter], :, ii + alter]
        )
        alter *= -1

        # Subtract residiual sky due to varying sky-brightness over obserations
        median = np.tile(
            np.nanmedian(flux_cube_out[kk, :, ii], axis=0),
            (flux_cube_out[kk, :, ii].shape[0], 1),
        )
        median[bpmap_cube_out[kk, :, ii].astype("bool")] = 0
        flux_cube_out[kk, :, ii] = flux_cube_out[kk, :, ii] - median

    # Form A-B - shifted(B-A) pairs
    alter = 1
    for ii, kk in enumerate(v_range):
        if alter == 1:
            flux_cube_out[:, :, ii] = (
                flux_cube_out[:, :, ii] + flux_cube_out[:, :, ii + 1]
            )
            error_cube_out[:, :, ii] = np.sqrt(
                error_cube_out[:, :, ii] ** 2.0 + error_cube_out[:, :, ii + 1] ** 2.0
            )
            bpmap_cube_out[:, :, ii] = (
                bpmap_cube_out[:, :, ii] + bpmap_cube_out[:, :, ii + 1]
            )
        elif alter == -1:
            flux_cube_out[:, :, ii] = np.nan
            error_cube_out[:, :, ii] = np.nan
            bpmap_cube_out[:, :, ii] = np.ones_like(bpmap_cube_out[:, :, ii]) * 666
        alter *= -1

    n_pix = np.ones_like(bpmap_cube_out) + (~(bpmap_cube_out.astype("bool"))).astype(
        "int"
    )
    flux_cube_out = flux_cube_out / n_pix
    error_cube_out = error_cube_out / (n_pix)

    good_mask = (bpmap_cube_out == 0) | (bpmap_cube_out == 10) | (bpmap_cube_out == 2)
    bpmap_cube_out[good_mask] = 0

    return flux_cube_out, error_cube_out, bpmap_cube_out


def filter_bad_values(wl, flux, error):
    medfilter = signal.medfilt(flux, 501)
    mask = np.logical_and(abs(flux - medfilter) < 3 * error, ~np.isnan(flux))
    f = interpolate.interp1d(
        wl[mask], flux[mask], bounds_error=False, fill_value=np.nan
    )
    g = interpolate.interp1d(
        wl[mask], error[mask], bounds_error=False, fill_value=np.nan
    )
    return wl, f(wl), g(wl)


class XshOrder2D(object):
    # Originally written by Johannes Zabl. Adapted and updated by Jonatan Selsing.
    def __init__(self, fname):
        self.fname = fname
        self.hdul = fits.open(fname)

        self.get_order_information()

    def get_order_information(self):
        # from astropy import wcs

        self.norders = int(len(self.hdul) / 3)

        header_list = [self.hdul[i * 3].header for i in range(self.norders)]
        wcs_list = [wcs.WCS(header) for header in header_list]

        self.cdelt1 = header_list[0]["CDELT1"]
        self.naxis1_list = [header["NAXIS1"] for header in header_list]
        self.naxis2 = header_list[0]["NAXIS2"]

        self.start_wave_list = [wcs.wcs_pix2world(1, 1, 1)[0] for wcs in wcs_list]

        self.end_wave_list = [
            wcs.wcs_pix2world(naxis1, 1, 1)[0]
            for wcs, naxis1 in zip(wcs_list, self.naxis1_list)
        ]
        self.wcs_list, self.header_list = wcs_list, header_list

    def create_out_frame_empty(self):
        self.npixel = (
            int((self.end_wave_list[0] - self.start_wave_list[-1]) / self.cdelt1) + 1
        )
        # print(self.cdelt1)
        self.ref_wcs = self.wcs_list[-1]
        self.data_new = np.zeros((self.naxis2, self.npixel))
        self.err_new = self.data_new.copy()
        self.qual_new = self.data_new.copy()

    def fill_data(self):
        weight_new = self.data_new.copy().astype(np.float64)
        pixels_new = np.zeros_like(weight_new)

        for i in range(self.norders):
            # In zero based coordinates
            start_pix = int(
                np.floor(self.ref_wcs.wcs_world2pix(self.start_wave_list[i], 0, 0)[0])
            )
            end_pix = int(
                np.floor(self.ref_wcs.wcs_world2pix(self.end_wave_list[i], 0, 0)[0])
            )

            data_inter = self.hdul[i * 3].data.astype(np.float64)
            err_inter = self.hdul[i * 3 + 1].data.astype(np.float64)
            qual_inter = self.hdul[i * 3 + 2].data
            mask = qual_inter > 0
            err_inter[mask] = 1e15

            err_cols = err_inter.copy() * 0.0 + np.median(err_inter, axis=0)
            err_cols[mask] = 1e15

            weight_inter = 1.0 / np.square(err_cols)

            self.err_new[:, start_pix : end_pix + 1] += (
                weight_inter**2.0 * err_inter**2.0
            )
            self.data_new[:, start_pix : end_pix + 1] += weight_inter * data_inter
            weight_new[:, start_pix : end_pix + 1] += weight_inter
            pixels_new[:, start_pix : end_pix + 1] += (~mask).astype("int")

        self.data_new /= weight_new
        self.err_new = np.sqrt(self.err_new / weight_new**2.0)
        self.qual_new = (self.err_new > 1e10).astype(int)

        self.data_new = self.data_new.astype(np.float32)
        self.err_new = self.err_new.astype(np.float32)

    def create_final_hdul(self):
        hdul_new = fits.HDUList()
        hdu_data = fits.PrimaryHDU(self.data_new, self.hdul[-3].header)
        hdu_err = fits.PrimaryHDU(self.err_new, self.hdul[-2].header)
        hdu_qual = fits.PrimaryHDU(self.qual_new, self.hdul[-1].header)

        hdu_data.header["EXTNAME"] = "FLUX"
        hdu_data.header["PIPEFILE"] = "MERGe_PYTHON_JZ"
        hdu_err.header["EXTNAME"] = "ERRS"
        hdu_err.header["SCIDATA"] = "FLUX"
        hdu_err.header["QUALDATA"] = "QUAL"
        hdu_qual.header["EXTNAME"] = "QUAL"
        hdu_qual.header["SCIDATA"] = "FLUX"
        hdu_qual.header["ERRDATA"] = "ERRS"

        hdul_new.append(hdu_data)
        hdul_new.append(hdu_err)
        hdul_new.append(hdu_qual)

        self.hdul_new = hdul_new

    def write_result(self, fname_out, clobber=False):
        self.hdul_new.writeto(fname_out, overwrite=True)

    def do_all(self, fname_out, clobber=False):
        self.create_out_frame_empty()
        self.fill_data()
        self.create_final_hdul()
        self.write_result(fname_out, clobber=clobber)


def ADC_corr_guess(header, waveaxis):
    # Corrections to slit position from broken ADC, taken DOI: 10.1086/131052
    # Pressure in hPa, Temperature in Celcius
    p, T = (
        header["HIERARCH ESO TEL AMBI PRES END"],
        header["HIERARCH ESO TEL AMBI TEMP"],
    )
    # Convert hPa to mmHg
    p = p * 0.7501
    # Wavelength in microns
    wl_m = waveaxis / 1e4
    # Refractive index in dry air (n - 1)1e6
    eq_1 = 64.328 + (29498.1 / (146 - wl_m**-2)) + (255.4 / (41 - wl_m**-2))
    # Corrections for ambient temperature and pressure
    eq_2 = eq_1 * (
        (p * (1.0 + (1.049 - 0.0157 * T) * 1e-6 * p)) / (720.883 * (1.0 + 0.003661 * T))
    )
    # Correction from water vapor. Water vapor obtained from the Antione equation, https://en.wikipedia.org/wiki/Antoine_equation
    eq_3 = eq_2 - ((0.0624 - 0.000680 * wl_m**-2) / (1.0 + 0.003661 * T)) * 10 ** (
        8.07131 - (1730.63 / (233.426 + T))
    )
    # Isolate n
    n = eq_3 / 1e6 + 1
    # Angle relative to zenith
    z = np.arccos(1 / header["HIERARCH ESO TEL AIRM START"])

    # Zero-deviation wavelength of arms, from http://www.eso.org/sci/facilities/paranal/instruments/xshooter/doc/VLT-MAN-ESO-14650-4942_v87.pdf
    if header["HIERARCH ESO SEQ ARM"] == "UVB":
        zdwl = 0.405
    elif header["HIERARCH ESO SEQ ARM"] == "VIS":
        zdwl = 0.633
    elif header["HIERARCH ESO SEQ ARM"] == "NIR":
        zdwl = 1.31
    else:
        raise ValueError(
            "Input image does not contain header keyword 'HIERARCH ESO SEQ ARM'. Cannot determine ADC correction."
        )

    zdwl_inx = find_nearest(wl_m, zdwl)

    # Direction of movement
    direction = 1

    # Correction of position on slit, relative to Zero-deviation wavelength
    dR = direction * (206265 * (n - n[zdwl_inx]) * np.tan(z))

    return dR


def slitcorr(header, waveaxis):
    # Theoretical slitloss based on DIMM seeing
    try:
        seeing = header["SEEING"]
    except:
        seeing = np.nanmin(
            header["HIERARCH ESO TEL AMBI FWHM START"],
            header["HIERARCH ESO TEL AMBI FWHM END"],
        )

    # Get slit width
    if header["HIERARCH ESO SEQ ARM"] == "UVB":
        slit_width = float(header["HIERARCH ESO INS OPTI3 NAME"].split("x")[0])
    elif header["HIERARCH ESO SEQ ARM"] == "VIS":
        slit_width = float(header["HIERARCH ESO INS OPTI4 NAME"].split("x")[0])
    elif header["HIERARCH ESO SEQ ARM"] == "NIR":
        slit_width = float(header["HIERARCH ESO INS OPTI5 NAME"].split("x")[0])

    # Correct seeing for airmass
    airmass = np.nanmean(
        [header["HIERARCH ESO TEL AIRM START"], header["HIERARCH ESO TEL AIRM END"]]
    )

    seeing_airmass_corr = seeing * (airmass) ** (3 / 5)

    # Theoretical wavelength dependence
    haxis_0 = 5000  # Å, DIMM center
    S0 = seeing_airmass_corr / haxis_0 ** (-1 / 5)
    seeing_theo = S0 * waveaxis ** (-1 / 5)

    # Calculating slit-losses based on 2D Moffat
    sl = [0] * len(seeing_theo)
    for ii, kk in enumerate(seeing_theo):
        sl[ii] = get_slitloss(kk, slit_width)
    slitcorr = np.array(sl)

    return slitcorr
