#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Plotting
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as pl
import seaborn

seaborn.set_style("ticks")
from matplotlib.backends.backend_pdf import PdfPages

# Additional imports
import glob
from astropy.io import fits
from astropy import wcs
from scipy.signal import medfilt2d
from scipy.ndimage.filters import median_filter
import copy
import numpy as np
from operator import or_
from scipy import optimize, interpolate, signal
from numpy.polynomial import chebyshev
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve
import numba

# Utility packages
from util import *


class XSHorder_object(object):
    """Object to contain a list of xshooter order 2D images"""

    def __init__(self, order_flist, merge_flist, sky_flist, sdp_flist):
        # Get lists to contain
        self.filelist = order_flist
        self.merge_filelist = merge_flist
        self.sky_filelist = sky_flist
        self.sdp_filelist = sdp_flist

        self.nfiles = len(order_flist)
        self.hdul = [fits.open(ii) for ii in self.filelist]
        self.merge_hdul = [fits.open(ii) for ii in self.merge_filelist]

        # Make reference hdus
        self.hdu = self.hdul[0]
        self.merge_hdu = self.merge_hdul[0]
        self.header = self.merge_hdu[0].header

        # Get order header information
        self.get_order_headers()

        # Get order wcs
        self.get_order_wcs()

        # Get arm
        self.arm = self.merge_hdu[0].header["HIERARCH ESO SEQ ARM"]

        # Get seeing

        # Theoretical slitloss based on DIMM seeing
        fwhm = np.nanmean(
            [
                np.nanmean(
                    [
                        ii[0].header["HIERARCH ESO TEL AMBI FWHM START"],
                        ii[0].header["HIERARCH ESO TEL AMBI FWHM END"],
                    ]
                )
                for ii in self.hdul
            ]
        )

        # Correct seeing for airmass
        airm = np.nanmean(
            [
                np.nanmean(
                    [
                        ii[0].header["HIERARCH ESO TEL AIRM START"],
                        ii[0].header["HIERARCH ESO TEL AIRM END"],
                    ]
                )
                for ii in self.hdul
            ]
        )

        self.seeing = fwhm * (airm) ** (3 / 5)
        self.seeing_pix = self.seeing / self.cdelt2

        # Get sky emission spectrum to finetune wavelenth array
        self.sky_hdul = [fits.open(ii) for ii in self.sky_filelist]

    def get_order_headers(self):
        # Get number of orders in image
        norders = [int(len(ii) / 3) for ii in self.hdul]

        # Images must have same number of orders
        if not norders.count(norders[0]) == len(norders):
            raise TypeError("Input order list does not have the same number of orders.")

        self.norders = norders[0]

        # Make a nested list with all order headers
        self.order_header_outer = [
            [kk[ll * 3].header for ll in range(self.norders)] for kk in self.hdul
        ]

        # Construct transposed list to loop through spatial dir
        self.order_header_outer_t = list(map(list, zip(*self.order_header_outer)))

    def get_order_wcs(self):
        # Construct wcs, using each order header for all files
        self.wcs_outer = [
            [wcs.WCS(header) for header in hdus] for hdus in self.order_header_outer
        ]

        # Construct transposed list to loop through spatial dir
        self.wcs_outer_t = list(map(list, zip(*self.wcs_outer)))

        # Find number of order elements
        self.naxis1_list = [header["NAXIS1"] for header in self.order_header_outer[0]]
        self.naxis2_list = [header["NAXIS2"] for header in self.order_header_outer[0]]

        # Images must have same number of spatial pixels
        if not self.naxis2_list.count(self.naxis2_list[0]) == len(self.naxis2_list):
            raise TypeError(
                "Input image list does not have the same number of spatial pixels."
            )

        self.naxis2 = self.naxis2_list[0]

        self.cdelt1_list = [header["CDELT1"] for header in self.order_header_outer[0]]
        self.cdelt2_list = [header["CDELT2"] for header in self.order_header_outer[0]]

        # Images must have same sampling in both directions
        if not self.cdelt1_list.count(self.cdelt1_list[0]) == len(self.cdelt1_list):
            raise TypeError(
                "Input image list does not have the same wavelength sampling."
            )
        if not self.cdelt2_list.count(self.cdelt2_list[0]) == len(self.cdelt2_list):
            raise TypeError("Input image list does not have the same spatial sampling.")

        self.cdelt1 = self.cdelt1_list[0]
        self.cdelt2 = self.cdelt2_list[0]

        # Get CRVAL1 for orders
        self.crval1_list = [header["CRVAL1"] for header in self.order_header_outer[0]]

        # Get wavelength extent of orders in nm
        self.start_wave_list = [
            wcs.wcs_pix2world(1, 1, 1)[0] + 0 for wcs in self.wcs_outer[0]
        ]
        self.end_wave_list = [
            wcs.wcs_pix2world(naxis1, 1, 1)[0] + 0
            for wcs, naxis1 in zip(self.wcs_outer[0], self.naxis1_list)
        ]

        # Spatial offset of individual images in arcsec
        self.cumoff = [
            header["HIERARCH ESO SEQ CUMOFF Y"]
            for header in self.order_header_outer_t[0]
        ]

        # Get spatial extent of orders by looping over the transposed list
        self.start_size_list = [
            wcs.wcs_pix2world(1, 1, 1)[1] + self.cumoff[elem]
            for elem, wcs in enumerate(self.wcs_outer_t[0])
        ]
        self.end_size_list = [
            wcs.wcs_pix2world(1, naxis2, 1)[1] + self.cumoff[elem]
            for elem, (wcs, naxis2) in enumerate(
                zip(self.wcs_outer_t[0], self.naxis2_list)
            )
        ]

        # Construct merged wavelength array
        self.waveaxis_merged = (
            (np.arange(self.header["NAXIS1"])) + 1 - self.header["CRPIX1"]
        ) * self.header["CDELT1"] + self.header["CRVAL1"]

    def make_empty_order_out_frame(self):
        # Get spatial offsets for total output image.
        max_neg_offset = min(self.start_size_list + self.end_size_list)
        max_pos_offset = max(self.start_size_list + self.end_size_list)

        # Get pixel size of output image
        self.nypixel = int(round((max_pos_offset - max_neg_offset) / self.cdelt2)) + 1

        # Pick refence wcvs
        self.ref_wcs = self.wcs_outer[0][-1]

        # Construct empty output arrays
        self.data_order_new = [
            np.zeros((self.nypixel, self.naxis1_list[ii]))
            for ii, order in enumerate(self.order_header_outer[0])
        ]
        self.err_order_new = copy.deepcopy(self.data_order_new)
        self.qual_order_new = copy.deepcopy(self.data_order_new)

        # Get physical axes
        self.waveaxis_list = [
            np.arange(self.naxis1_list[nn]) * self.cdelt1 + self.crval1_list[nn]
            for nn in range(self.norders)
        ]
        self.spataxis_list = [
            np.arange(self.nypixel) * self.cdelt2
            + self.header["CRVAL2"]
            - 2 * self.cumoff[0]
            for nn in range(self.norders)
        ]

    @numba.jit(parallel=True)
    def make_clean_mask(self):
        # Mask outliers in the direction of the stack and "bad" pixels.

        # Loop through orders
        for nn in numba.prange(self.norders):
            # Make storage cubes
            shp = np.shape(self.data_order_new[nn])
            data_store_cube = np.zeros((shp[0], shp[1], self.nfiles))
            err_store_cube = np.zeros((shp[0], shp[1], self.nfiles))
            qual_store_cube = np.zeros((shp[0], shp[1], self.nfiles))

            # Populate cube
            for ii in range(self.nfiles):
                # In zero based coordinates
                start_pix = int(
                    round(
                        np.floor(
                            self.ref_wcs.wcs_world2pix(0, self.start_size_list[ii], 0)[
                                1
                            ]
                        )
                    )
                )
                end_pix = int(
                    round(
                        np.floor(
                            self.ref_wcs.wcs_world2pix(0, self.end_size_list[ii], 0)[1]
                        )
                    )
                )

                # Get offset to start at pixel 0
                min_pix = abs(min([start_pix, end_pix]))

                # Define array slice
                cut_slice = slice(start_pix + min_pix, end_pix + min_pix + 1)

                # Single image input
                data_temp = self.hdul[ii][nn * 3].data / 1e-17
                err_temp = self.hdul[ii][nn * 3 + 1].data / 1e-17
                qual_temp = self.hdul[ii][nn * 3 + 2].data

                data_store_cube[cut_slice, :, ii] = data_temp
                err_store_cube[cut_slice, :, ii] = err_temp
                qual_store_cube[cut_slice, :, ii] = qual_temp

            # Mask 3(5)-sigma outliers in the direction of the stack
            m, s = (
                np.ma.median(
                    np.ma.array(data_store_cube, mask=qual_store_cube), axis=2
                ).data,
                np.std(np.ma.array(data_store_cube, mask=qual_store_cube), axis=2).data,
            )
            if self.arm == "NIR":
                sigma_mask = 5
            else:
                sigma_mask = 3
            l, h = (
                np.tile((m - sigma_mask * s).T, (self.nfiles, 1, 1)).T,
                np.tile((m + sigma_mask * s).T, (self.nfiles, 1, 1)).T,
            )
            qual_store_cube[(data_store_cube < l) | (data_store_cube > h)] = 666

            # Introduce filter based on smoothed image
            flux_filt, error_filt, bp_filt = (
                np.zeros_like(data_store_cube),
                np.zeros_like(data_store_cube),
                np.zeros_like(data_store_cube),
            )
            for ii in range(data_store_cube.shape[2]):
                flux_filt[:, :, ii] = medfilt2d(data_store_cube[:, :, ii], 3)
                error_filt[:, :, ii] = medfilt2d(err_store_cube[:, :, ii], 3)

            bp_filt = abs(flux_filt - data_store_cube) > 7 * error_filt
            bp_filt[
                int(np.floor(self.nypixel / 2 - 2)) : int(
                    np.ceil(self.nypixel / 2 + 2)
                ),
                :,
                :,
            ] = False
            qual_store_cube[bp_filt == True] = 555

            # Probagate the updated bad pixel maps to the individual orders
            for ii in range(self.nfiles):
                # In zero based coordinates
                start_pix = int(
                    round(
                        np.floor(
                            self.ref_wcs.wcs_world2pix(0, self.start_size_list[ii], 0)[
                                1
                            ]
                        )
                    )
                )
                end_pix = int(
                    round(
                        np.floor(
                            self.ref_wcs.wcs_world2pix(0, self.end_size_list[ii], 0)[1]
                        )
                    )
                )

                # Get offset to start at pixel 0
                min_pix = abs(min([start_pix, end_pix]))

                # Define array slice
                cut_slice = slice(start_pix + min_pix, end_pix + min_pix + 1)

                # Make mask based on bpmaps
                mask = (qual_store_cube > 0).astype("int")
                self.hdul[ii][kk * 3 + 2].data = mask[cut_slice, :, ii]

    @numba.jit(parallel=True)
    def merge_images(self):
        weight_map = copy.deepcopy(self.data_order_new)
        count_map = copy.deepcopy(self.data_order_new)

        for nn in numba.prange(self.norders):
            if self.clean:
                self.make_clean_mask()

            if self.NOD:
                self.data_order_new[nn] = self.hdul[0][nn * 3].data / 1e-17
                self.err_order_new[nn] = self.hdul[0][nn * 3 + 1].data / 1e-17
                self.qual_order_new[nn] = self.hdul[0][nn * 3 + 2].data
                continue

            for ii in range(self.nfiles):
                # In zero based coordinates
                start_pix = int(
                    round(
                        np.floor(
                            self.ref_wcs.wcs_world2pix(0, self.start_size_list[ii], 0)[
                                1
                            ]
                        )
                    )
                )
                end_pix = int(
                    round(
                        np.floor(
                            self.ref_wcs.wcs_world2pix(0, self.end_size_list[ii], 0)[1]
                        )
                    )
                )

                # Get offset to start at pixel 0
                min_pix = abs(min([start_pix, end_pix]))

                # Define array slice
                cut_slice = slice(start_pix + min_pix, end_pix + min_pix + 1)

                # Single image input
                data_temp = self.hdul[ii][nn * 3].data / 1e-17
                err_temp = self.hdul[ii][nn * 3 + 1].data / 1e-17
                qual_temp = self.hdul[ii][nn * 3 + 2].data

                # Make boolean mask based on bad pixel map
                mask = qual_temp > 0

                # Mask 2 pixels around the edges
                mask[0, :] = True
                mask[-1, :] = True
                # Use mask to construct masked arrays
                data_temp[mask] = 0
                err_temp[mask] = np.nan

                # Use a smoothed version of the noise map as weights, where nan are inpainted. Skip if all entries are masked
                if np.isnan(copy.deepcopy(err_temp)).all():
                    err_smooth = err_temp
                else:
                    run_var = np.ones(self.naxis1)
                    for kk in numba.prange(self.naxis1):
                        err_bin = (
                            1
                            / (
                                err_temp[:, kk - 4 : kk + 4][
                                    ~(mask[:, kk - 4 : kk + 4])
                                ]
                            )
                            ** 2
                        )
                        if len(err_bin) != 0:
                            run_var[kk] = np.median(err_bin.flatten())
                    err_smooth_tiled = np.tile(
                        np.nanmedian(run_var, axis=0), (self.naxis2, 1)
                    )

                    # err_smooth_tiled = np.tile(np.nanmedian(err_temp, axis=0), (self.naxis2, 1))
                    err_temp[mask] = err_smooth_tiled[mask]
                    # err_smooth = median_filter(err_temp, size = 10, mode="reflect")
                    err_smooth = err_temp

                # # Make weight map based on background variance in boxcar window
                # shp = error_cube.shape
                # weight_cube = np.zeros_like(error_cube)
                # if not same:
                #     for ii in range(shp[2]):
                #         run_var = np.ones(shp[1])
                #         for kk in np.arange(shp[1]):
                #             err_bin = 1/(error_cube[:, kk-4:kk+4, ii][~(bpmap_cube[:, kk-4:kk+4, ii].astype("bool"))])**2
                #             if len(err_bin) != 0:
                #                 run_var[kk] = np.median(err_bin.flatten())
                #         weight_cube[:, :, ii] = np.tile(run_var, (shp[0], 1))
                # elif same:
                #     for ii in range(shp[2]):
                #         weight_cube[:, :, ii] = np.tile(np.median(1/(error_cube[~(bpmap_cube.astype("bool"))])**2), (shp[0], shp[1]))

                # # Normlize weights
                # weight_cube[mask_cube] = 0
                # weight_cube = weight_cube/np.tile(np.sum(weight_cube, axis=2).T, (shp[2], 1, 1)).T

                err_temp[mask] = 0

                # Smoothed inverse varience weights
                # weight_temp = np.ones_like(err_temp)
                weight_temp = 1.0 / err_smooth**2.0

                weight_temp[mask] = 0

                # Construct count images to keep track of the number of contributing pixels
                count_img = np.ones_like(data_temp)

                # Iteratively add the images
                self.data_order_new[nn][cut_slice, :] += weight_temp * data_temp
                self.err_order_new[nn][cut_slice, :] += (
                    weight_temp**2.0 * err_temp**2.0
                )
                self.qual_order_new[nn][cut_slice, :] += mask.astype("int")
                weight_map[nn][cut_slice, :] += weight_temp
                count_map[nn][cut_slice, :] += count_img

            # Scale by the summed weights.
            data_scaled = self.data_order_new[nn] / weight_map[nn]
            err_scaled = np.sqrt(self.err_order_new[nn] / weight_map[nn] ** 2.0)

            # Mask pixels in out map, where all input values are masked
            filter_mask = count_map[nn] == self.qual_order_new[nn]

            # Replace masked value with nan
            data_scaled[filter_mask] = np.nan
            err_scaled[filter_mask] = np.nan
            self.qual_order_new[nn][~filter_mask] = 0
            self.qual_order_new[nn][filter_mask] = 1

            # Store in order lists
            self.data_order_new[nn] = list(data_scaled)
            self.err_order_new[nn] = list(err_scaled)
            self.qual_order_new[nn] = list(self.qual_order_new[nn])

    def write_order_fits(self):
        """
        Method to save combined image
        """

        for ii, kk in enumerate(self.hdu):
            count = int(np.floor(ii / 3))
            self.hdu[count * 3].data = self.data_order_new[count]
            self.hdu[count * 3 + 1].data = self.err_order_new[count]
            self.hdu[count * 3 + 2].data = self.qual_order_new[count]

            # Update WCS
            self.hdu[ii].header["CRVAL2"] = (
                self.hdu[ii].header["CRVAL2"] - 2 * self.cumoff[0]
            )

        # Write file
        self.hdu.writeto(self.base_fname_out + "_orders.fits", overwrite=True)

    def make_empty_binned_order_frame(self):
        # Get the total wavelength size in pixels
        self.bin_nxpixel = (
            np.floor(
                (np.array(self.end_wave_list) - np.array(self.start_wave_list))
                / (self.cdelt1 * self.bin_factor)
            )
        ).astype("int") + 1
        # Pick refence wcvs
        self.ref_wcs = self.wcs_outer[0][-1]

        # Construct empty output arrays
        self.data_order_binned_new = [
            np.zeros((self.naxis2, self.bin_nxpixel[ii]))
            for ii, order in enumerate(self.order_header_outer[0])
        ]
        self.err_order_binned_new = copy.deepcopy(self.data_order_binned_new)
        self.qual_order_binned_new = copy.deepcopy(self.data_order_binned_new)

    @numba.jit(parallel=True)
    def bin_orders(self):
        # Loop through the orders
        for nn in numba.prange(self.norders):
            # Make binned spectrum
            bin_flux, bin_error, bin_bpmap = bin_image(
                self.hdu[nn * 3].data,
                self.hdu[nn * 3 + 1].data,
                self.hdu[nn * 3 + 2].data,
                self.bin_factor,
                weight=False,
            )

            # Populate empty frame
            self.data_order_binned_new[nn] = bin_flux
            self.err_order_binned_new[nn] = bin_error
            self.qual_order_binned_new[nn] = bin_bpmap

    def write_binned_fits(self):
        for ii, kk in enumerate(self.hdu):
            count = int(np.floor(ii / 3))
            self.hdu[count * 3].data = self.data_order_binned_new[count]
            self.hdu[count * 3 + 1].data = self.err_order_binned_new[count]
            self.hdu[count * 3 + 2].data = self.qual_order_binned_new[count]

            # Update WCS
            self.hdu[ii].header["CDELT1"] = (
                self.hdu[ii].header["CDELT1"] * self.bin_factor
            )
            self.hdu[ii].header["CD1_1"] = (
                self.hdu[ii].header["CD1_1"] * self.bin_factor
            )

        # Write file
        self.hdu.writeto(self.base_fname_out + "_orders_binned.fits", overwrite=True)

        for ii, kk in enumerate(self.hdu):
            count = int(np.floor(ii / 3))
            self.hdu[count * 3].data = self.data_order_new[count]
            self.hdu[count * 3 + 1].data = self.err_order_new[count]
            self.hdu[count * 3 + 2].data = self.qual_order_new[count]

            # Update WCS
            self.hdu[ii].header["CDELT1"] = (
                self.hdu[ii].header["CDELT1"] / self.bin_factor
            )
            self.hdu[ii].header["CD1_1"] = (
                self.hdu[ii].header["CD1_1"] * self.bin_factor
            )

    # @numba.jit(parallel=True)
    def subtract_residual_sky(self):
        self.sky_background = copy.deepcopy(self.data_order_new)
        self.sky_background_error = copy.deepcopy(self.data_order_new)

        pp = PdfPages(self.base_fname_out + "sky_subtractions.pdf")

        # Make trace mask

        trace_offsets = np.array(self.trace_mask) / self.cdelt2
        traces = []
        for ii in trace_offsets:
            traces.append(self.nypixel / 2 + ii)

        # Masking pixels in frame.
        trace_mask = np.zeros(self.nypixel).astype("bool")
        for ii, kk in enumerate(traces):
            trace_mask[int(kk - self.seeing_pix) : int(kk + self.seeing_pix)] = True

        for nn in range(self.norders):
            flux = np.array(self.data_order_new[nn])
            error = np.array(self.err_order_new[nn])
            bpmap = np.array(self.qual_order_new[nn])
            sky_back_temp = np.zeros_like(self.data_order_new[nn])

            full_trace_mask = np.tile(trace_mask, (self.naxis1_list[nn], 1)).T
            full_mask = bpmap.astype("bool") | full_trace_mask

            for ii in numba.prange(len(self.waveaxis_list[nn])):
                # Pick mask slice
                mask = full_mask[:, ii]

                # Sigma clip before sky-estimate to remove noisy pixels with bad error estimate.
                m, s = np.nanmean(flux[:, ii]), np.nanstd(flux[:, ii])
                clip_mask = (flux[:, ii] < m - s) | (flux[:, ii] > m + s)

                # Combine masks
                mask = mask | clip_mask

                # Subtract polynomial estiamte of sky
                vals = flux[:, ii][~mask]
                errs = error[:, ii][~mask]

                try:
                    chebfit = chebyshev.chebfit(
                        self.spataxis_list[nn][~mask], vals, deg=2, w=1 / errs
                    )
                    # chebfit = chebyshev.chebfit(self.spataxis_list[nn][~mask], vals, deg = 2)
                    chebfitval = chebyshev.chebval(self.spataxis_list[nn], chebfit)
                    # chebfitval[chebfitval <= 0] = 0
                except TypeError:
                    print(
                        "Empty array for sky-estimate at index "
                        + str(ii)
                        + ". Sky estimate replaced with zeroes."
                    )
                    chebfitval = np.zeros_like(self.spataxis_list[nn])
                except:
                    print(
                        "Polynomial fit did not converge at index "
                        + str(ii)
                        + ". Sky estimate replaced with median value."
                    )
                    chebfitval = np.ones_like(self.spataxis_list[nn]) * np.ma.median(
                        self.spataxis_list[nn][~mask]
                    )

                if ii % 100 == 0 and ii != 0:
                    # Plotting for quality control
                    pl.errorbar(
                        self.spataxis_list[nn][~mask],
                        vals,
                        yerr=errs,
                        fmt=".k",
                        capsize=0,
                        elinewidth=0.5,
                        ms=3,
                    )
                    pl.plot(self.spataxis_list[nn], chebfitval)
                    pl.xlabel("Spatial index")
                    pl.ylabel("Flux density")
                    pl.title(
                        "Quality test: Sky estimate at index: "
                        + str(ii)
                        + " for order: "
                        + str(nn)
                    )
                    pp.savefig()
                    pl.clf()
                    # pl.show()

                sky_back_temp[:, ii] = chebfitval

            self.sky_background[nn] = convolve(sky_back_temp, Gaussian2DKernel(1.0))

            self.data_order_new[nn] -= self.sky_background[nn]
            self.hdu[nn * 3].data -= self.sky_background[nn]
        pp.close()

    @numba.jit(parallel=True)
    def finetune_wavlength_solution(self):
        print("")
        print(
            "Cross correlating with synthetic sky to obtain refinement to wavlength solution ..."
        )
        print("")
        self.correction_factor = [0] * self.nfiles

        for nn in numba.prange(self.nfiles):
            em_sky = self.sky_hdul[nn][0].data
            haxis = self.waveaxis_merged

            # Remove continuum
            mask = ~np.isnan(em_sky) & (haxis < 1800) & (haxis > 350)
            hist, edges = np.histogram(em_sky[mask], bins="auto")
            max_idx = find_nearest(hist, max(hist))
            sky = em_sky - edges[max_idx]
            mask = ~np.isnan(sky) & (sky > 0) & (haxis < 1800) & (haxis > 350)

            # Load synthetic sky
            sky_model = fits.open("statics/skytable_hres.fits")
            wl_sky = 1e4 * (sky_model[1].data.field("lam"))  # In micron
            flux_sky = sky_model[1].data.field("flux")

            # Convolve to observed grid
            from scipy.interpolate import interp1d

            f = interp1d(
                wl_sky,
                convolve(sky_model[1].data.field("flux"), Gaussian1DKernel(stddev=10)),
                bounds_error=False,
                fill_value=np.nan,
            )
            synth_sky = f(haxis)

            # Cross correlate with redshifted spectrum and find velocity offset
            offsets = np.arange(-0.0005, 0.0005, 0.00001)
            correlation = np.zeros(offsets.shape)
            for ii, kk in enumerate(offsets):
                synth_sky = f(haxis * (1.0 + kk))
                correlation[ii] = np.correlate(
                    sky[mask] * (np.nanmax(synth_sky) / np.nanmax(sky)), synth_sky[mask]
                )

            # Index with maximal value
            max_idx = find_nearest(correlation, max(correlation))
            # Corrections to apply to original spectrum, which maximizes correlation.
            self.correction_factor[nn] = 1.0 + offsets[max_idx]
            print(
                "Found preliminary velocity offset: "
                + str((self.correction_factor[nn] - 1.0) * 3e5)
                + " km/s"
            )
            print("")
            print(
                "Minimising residuals between observed sky and convolved synthetic sky to obtain the sky PSF ..."
            )
            print("")

            # Zero-deviation wavelength of arms, from http://www.eso.org/sci/facilities/paranal/instruments/xshooter/doc/VLT-MAN-ESO-14650-4942_v87.pdf
            if self.header["HIERARCH ESO SEQ ARM"] == "UVB":
                zdwl = 4050
                pixel_width = 50
            elif self.header["HIERARCH ESO SEQ ARM"] == "VIS":
                zdwl = 6330
                pixel_width = 50
            elif self.header["HIERARCH ESO SEQ ARM"] == "NIR":
                zdwl = 13100
                pixel_width = 50

            # Get seeing PSF by minimizing the residuals with a theoretical sky model, convovled with an increasing psf.
            psf_width = np.arange(1, pixel_width, 1)
            res = np.zeros(psf_width.shape)
            for ii, kk in enumerate(psf_width):
                # Convolve sythetic sky with Gaussian psf
                convolution = convolve(flux_sky, Gaussian1DKernel(stddev=kk))
                # Interpolate high-res syntheric sky onto observed wavelength grid.
                f = interp1d(wl_sky, convolution, bounds_error=False, fill_value=np.nan)
                synth_sky = f(haxis * self.correction_factor[nn])
                # Calculate squared residuals
                residual = np.nansum(
                    (
                        synth_sky[mask]
                        * (np.nanmax(sky[mask]) / np.nanmax(synth_sky[mask]))
                        - sky[mask]
                    )
                    ** 2.0
                )
                res[ii] = residual

            # Index of minimal residual
            min_idx = find_nearest(res, min(res))

            # Wavelegth step corresponding psf width in FWHM
            R, seeing = np.zeros(psf_width.shape), np.zeros(psf_width.shape)
            for ii, kk in enumerate(psf_width):
                dlambda = np.diff(wl_sky[::kk]) * 2 * np.sqrt(2 * np.log(2))
                # Interpolate to wavelegth grid
                f = interp1d(
                    wl_sky[::kk][:-1], dlambda, bounds_error=False, fill_value=np.nan
                )
                dlambda = f(haxis)

                # PSF FWHM in pixels
                d_pix = dlambda / (10 * self.header["CDELT1"])
                # Corresponding seeing PSF FWHM in arcsec
                spatial_psf = d_pix * self.header["CDELT2"]

                # Index of zero-deviation
                zd_idx = find_nearest(haxis, zdwl)

                # Resolution at zero-deviation wavelength
                R[ii] = (haxis / dlambda)[zd_idx]
                # Seeing at zero-deviation wavelength
                seeing[ii] = spatial_psf[zd_idx]

            fig, ax1 = pl.subplots()
            ax1.yaxis.set_major_formatter(pl.NullFormatter())

            convolution = convolve(
                flux_sky, Gaussian1DKernel(stddev=psf_width[min_idx])
            )
            f = interp1d(wl_sky, convolution, bounds_error=False, fill_value=np.nan)
            synth_sky = f(haxis)

            # Cross correlate with redshifted spectrum and find velocity offset
            offsets = np.arange(-0.0005, 0.0005, 0.000001)
            correlation = np.zeros_like(offsets)
            for ii, kk in enumerate(offsets):
                synth_sky = f(haxis * (1.0 + kk))
                correlation[ii] = np.correlate(
                    sky[mask] * (np.nanmax(synth_sky[mask]) / np.nanmax(sky[mask])),
                    synth_sky[mask],
                )

            # Smooth cross-correlation
            correlation = convolve(correlation, Gaussian1DKernel(stddev=20))

            # Index with maximum correlation
            max_idx = find_nearest(correlation, max(correlation))
            self.correction_factor[nn] = 1.0 + offsets[max_idx]
            self.header["WAVECORR"] = self.correction_factor[nn]
            print(
                "Found refined velocity offset: "
                + str((self.correction_factor[nn] - 1.0) * 3e5)
                + " km/s"
            )
            print("")

            # Mask flux with > 3-sigma sky brightness
            self.sky_mask = f(haxis * self.correction_factor[nn]) > np.percentile(
                f(haxis * self.correction_factor), 99
            )
            ax2 = ax1.twiny()

            ax2.errorbar(
                offsets[max_idx] * 3e5,
                max(correlation) * (max(res) / max(correlation)),
                fmt=".k",
                capsize=0,
                elinewidth=0.5,
                ms=13,
                label="Wavelength correction:"
                + str(np.around((self.correction_factor - 1.0) * 3e5, decimals=1))
                + " km/s",
                color="r",
            )
            ax2.plot(
                offsets * 3e5, correlation * (max(res) / max(correlation)), color="r"
            )
            ax2.set_xlabel("Offset velocity / [km/s]", color="r")
            ax2.set_ylabel("Cross correlation", color="r")
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.set_major_formatter(pl.NullFormatter())
            ax2.legend(loc=2)
            pl.savefig(self.base_name + "Wavelength_cal.pdf")
            pl.clf()

    # @numba.jit(parallel=True)
    def fit_order_spsf(self):
        self.trace_model = copy.deepcopy(self.data_order_new)
        self.full_profile = copy.deepcopy(self.data_order_new)

        lower_element_nr = 1
        upper_element_nr = 1

        # Define edges of image for nodding to rid of negative traces
        if self.NOD:
            width = int(len(self.nypixel) / 3)
        else:
            width = 5

        pp = PdfPages(self.base_fname_out + "Quality_test_SPSF_fit_order.pdf")
        ppq = PdfPages(self.base_fname_out + "PSF_quality_control_order.pdf")

        for nn in range(self.norders):
            flux = self.data_order_binned_new[nn]
            error = self.err_order_binned_new[nn]

            # waveaxis = np.linspace(self.start_wave_list[nn], self.end_wave_list[nn], self.bin_nxpixel[nn])
            xlen = np.shape(flux)[1]
            waveaxis = (
                np.arange(xlen) * self.cdelt1 * self.bin_factor + self.crval1_list[nn]
            )
            full_waveaxis = (
                np.arange(self.naxis1_list[nn]) * self.cdelt1 + self.crval1_list[nn]
            )

            ylen = np.shape(flux)[0]
            spataxis = (
                np.arange(ylen) * self.cdelt2
                + self.header["CRVAL2"]
                - 2 * self.cumoff[0]
            )

            # Inital parameter guess
            fwhm_sigma = 2.0 * np.sqrt(
                2.0 * np.log(2.0)
            )  # Conversion between header seeing value and fit seeing value.
            if self.p0 == None:
                self.p0 = [
                    abs(1e1 * np.nanmean(flux[flux > 0])),
                    np.median(self.nypixel),
                    abs(self.header["HIERARCH ESO TEL AMBI FWHM START"]),
                    0,
                    0,
                ]
                if two_comp:
                    self.p0 = [
                        1e1 * np.nanmean(flux[flux > 0]),
                        np.median(self.nypixel),
                        abs(self.header["HIERARCH ESO TEL AMBI FWHM START"]),
                        0,
                        0,
                        5e-1 * np.nanmean(flux[flux > 0]),
                        np.median(self.nypixel) + 2,
                        0.5,
                        0.1,
                    ]

            if self.broken_adc:
                dR = ADC_corr_guess()

            # Parameter containers
            amp, cen, fwhm = (
                np.zeros_like(waveaxis),
                np.zeros_like(waveaxis),
                np.zeros_like(waveaxis),
            )
            eamp, ecen, efwhm = (
                np.zeros_like(waveaxis),
                np.zeros_like(waveaxis),
                np.zeros_like(waveaxis),
            )

            # Loop though along dispersion axis in the binned image and fit a Voigt
            x = np.arange(
                min(spataxis[width:-width]), max(spataxis[width:-width]), 0.01
            )
            inp_cent = self.p0[1]
            # fig, ax1 = pl.subplots(figsize=(5, 5))

            for ii in range(len(waveaxis)):
                try:
                    # Edit trace position guess by analytic ADC-amount
                    if self.broken_adc:
                        self.p0[1] = inp_cent + dR[ii]
                    elif not self.broken_adc:
                        self.p0[1] = inp_cent
                    # Fit SPSF
                    if self.two_comp:
                        popt, pcov = optimize.curve_fit(
                            Two_Moffat1D,
                            spataxis[width:-width],
                            flux[:, ii][width:-width],
                            p0=self.p0,
                            maxfev=5000,
                        )
                    elif not self.two_comp:
                        popt, pcov = optimize.curve_fit(
                            Moffat1D,
                            spataxis[width:-width],
                            flux[:, ii][width:-width],
                            p0=self.p0,
                            maxfev=5000,
                        )

                    pl.errorbar(
                        spataxis[width:-width],
                        flux[:, ii][width:-width],
                        yerr=error[:, ii][width:-width],
                        fmt=".k",
                        capsize=0,
                        elinewidth=0.5,
                        ms=3,
                    )
                    if self.two_comp:
                        pl.plot(x, Two_Moffat1D(x, *popt), label="Best-fit")
                    elif not self.two_comp:
                        pl.plot(x, Moffat1D(x, *popt), label="Best-fit")
                    guess_par = [popt[0]] + self.p0[1:]
                    # guess_par[4] = popt[4]
                    # guess_par[5] = popt[5]
                    if self.two_comp:
                        guess_par[-1] = popt[-1]
                        pl.plot(
                            x, Two_Moffat1D(x, *guess_par), label="Fit guess parameters"
                        )
                    elif not self.two_comp:
                        pl.plot(
                            x, Moffat1D(x, *guess_par), label="Fit guess parameters"
                        )
                    pl.title(
                        "Profile fit in order: " + str(nn) + " at index: " + str(ii)
                    )
                    pl.xlabel("Slit position / [arcsec]")
                    pl.ylabel("Flux density")
                    pl.legend()
                    pp.savefig()
                    pl.clf()

                except:
                    print(
                        "Fitting error at binned image index: "
                        + str(ii)
                        + ". Replacing fit value with guess and set fit error to 10^10"
                    )
                    popt, pcov = self.p0, np.diag(1e10 * np.ones_like(self.p0))
                amp[ii], cen[ii], fwhm[ii] = popt[0], popt[1], popt[2]
                eamp[ii], ecen[ii], efwhm[ii] = (
                    np.sqrt(np.diag(pcov)[0]),
                    np.sqrt(np.diag(pcov)[1]),
                    np.sqrt(np.diag(pcov)[2]),
                )

            # Mask edges
            ecen[:lower_element_nr] = 1e10
            ecen[-upper_element_nr:] = 1e10
            # Mask elements too close to guess, indicating a bad fit.
            ecen[
                abs(cen / ecen)
                > abs(np.nanmean(cen / ecen)) + 5 * np.nanstd(cen / ecen)
            ] = 1e10
            ecen[abs(amp - self.p0[0]) < self.p0[0] / 100] = 1e10
            ecen[abs(cen - self.p0[1]) < self.p0[1] / 100] = 1e10
            ecen[abs(fwhm - self.p0[2]) < self.p0[2] / 100] = 1e10

            # Remove the 5 highest S/N pixels
            ecen[np.argsort(fwhm / efwhm)[-3:]] = 1e10

            # Fit polynomial for center and iteratively reject outliers
            std_resid = 5
            while std_resid > 0.5:
                idx = np.isfinite(cen) & np.isfinite(ecen)
                fitcen = chebyshev.chebfit(
                    waveaxis[idx], cen[idx], deg=self.pol_degree[0], w=1 / ecen[idx]
                )
                resid = cen - chebyshev.chebval(waveaxis, fitcen)
                avd_resid, std_resid = np.median(resid[ecen != 1e10]), np.std(
                    resid[ecen != 1e10]
                )
                mask = (resid < avd_resid - std_resid) | (resid > avd_resid + std_resid)
                ecen[mask] = 1e10
            fitcenval = chebyshev.chebval(full_waveaxis, fitcen)
            # Plotting for quality control
            fig, (ax1, ax2, ax3) = pl.subplots(3, 1, figsize=(14, 14), sharex=True)

            ax1.errorbar(
                waveaxis, cen, yerr=ecen, fmt=".k", capsize=0, elinewidth=0.5, ms=7
            )
            ax1.plot(full_waveaxis, fitcenval)
            vaxis_range = max(spataxis) - min(spataxis)
            ax1.set_ylim((min(spataxis[width:-width]), max(spataxis[width:-width])))
            ax1.set_ylabel("Profile center / [arcsec]")
            ax1.set_title("Quality test: Center estimate")
            # Sigmama-clip outliers in S/N-space
            efwhm[ecen == 1e10] = 1e10
            efwhm[fwhm < 0.01] = 1e10
            efwhm[np.isnan(efwhm)] = 1e10

            fitfwhm = chebyshev.chebfit(
                waveaxis, fwhm, deg=self.pol_degree[1], w=1 / efwhm
            )
            fitfwhmval = chebyshev.chebval(full_waveaxis, fitfwhm)
            # Ensure positivity
            fitfwhmval[fitfwhmval < 0.1] = 0.1

            # Plotting for quality control
            ax2.errorbar(
                waveaxis, fwhm, yerr=efwhm, fmt=".k", capsize=0, elinewidth=0.5, ms=7
            )
            ax2.plot(full_waveaxis, fitfwhmval)
            ax2.set_ylim((0, 3))
            ax2.set_ylabel("Profile FWHM width / [arcsec]")
            ax2.set_title("Quality test: Profile width estimate")

            # Amplitude replaced with ones

            eamp[ecen == 1e10] = 1e10
            amp[amp < 0] = 1e-20
            amp = signal.medfilt(amp, 5)
            mask = ~(eamp == 1e10)
            f = interpolate.interp1d(
                waveaxis[mask], amp[mask], bounds_error=False, fill_value="extrapolate"
            )
            fitampval = f(full_waveaxis)
            fitampval[fitampval <= 0] = 1e-20  # np.nanmean(fitampval[fitampval > 0])

            # Plotting for quality control
            ax3.errorbar(waveaxis, amp, fmt=".k", capsize=0, elinewidth=0.5, ms=5)
            ax3.plot(full_waveaxis, fitampval)
            ax3.set_ylabel("Profile amplitude")
            ax3.set_title("Quality test: Profile amplitude estimate")
            ax3.set_xlabel(r"Wavelength / [$\mathrm{\AA}$]")
            pl.title("Quality test for order: %s" % str(nn))
            fig.subplots_adjust(hspace=0)
            ppq.savefig()
            # fig.savefig(self.base_fname_out + "PSF_quality_control.pdf")
            pl.close()

            self.full_profile[nn], self.trace_model[nn] = np.zeros_like(
                self.data_order_new[nn]
            ), np.zeros_like(self.data_order_new[nn])
            for ii, kk in enumerate(full_waveaxis):
                self.trace_model[nn][:, ii] = Moffat1D(
                    spataxis, fitampval[ii], fitcenval[ii], fitfwhmval[ii]
                )
                self.full_profile[nn][:, ii] = self.trace_model[nn][:, ii] / abs(
                    np.trapz(self.trace_model[nn][:, ii])
                )

        pp.close()
        ppq.close()

    @numba.jit(parallel=True)
    def extract_order_spectra(self):
        merged_optimal_spectrum = np.zeros(self.nxpixel)
        merged_optimal_error_spectrum = np.zeros(self.nxpixel)
        merged_optimal_bpmap = np.zeros(self.nxpixel)
        merged_optimal_weight_map = np.zeros(self.nxpixel)

        merged_standard_spectrum = np.zeros(self.nxpixel)
        merged_standard_error_spectrum = np.zeros(self.nxpixel)
        merged_standard_bpmap = np.zeros(self.nxpixel)
        merged_standard_weight_map = np.zeros(self.nxpixel)

        for nn in numba.prange(self.norders):
            # In zero based coordinates
            start_pix = int(
                np.floor(self.ref_wcs.wcs_world2pix(self.start_wave_list[nn], 0, 0)[0])
            )
            end_pix = int(
                np.floor(self.ref_wcs.wcs_world2pix(self.end_wave_list[nn], 0, 0)[0])
            )

            # Define array slice
            cut_slice = slice(start_pix, end_pix + 1)

            # Single image input
            data_temp = self.hdu[nn * 3].data
            err_temp = self.hdu[nn * 3 + 1].data
            qual_temp = self.hdu[nn * 3 + 2].data

            # Make boolean mask based on bad pixel map
            mask = qual_temp > 0

            # Use mask to construct masked arrays
            data_temp[mask] = np.nan
            err_temp[mask] = np.nan

            # Replace error image with median variance estimate to avoid including pixel-based weights
            bg_variance = signal.medfilt(
                np.tile(np.nanmedian(err_temp**2.0, axis=0), (self.nypixel, 1)),
                [1, 11],
            )

            # Get first extractions
            denom = np.nansum((self.full_profile[nn] ** 2.0 / bg_variance), axis=0)
            spectrum = (
                np.nansum(self.full_profile[nn] * data_temp / bg_variance, axis=0)
                / denom
            )
            errorspectrum_syn = np.sqrt(1 / denom)

            # Create synthetic variance based on error spectrum and profile
            syn_variance = (
                np.tile(errorspectrum_syn**2, (self.nypixel, 1))
                * self.full_profile[nn]
                + bg_variance
            )

            # Repeat extraction
            denom = np.nansum((self.full_profile[nn] ** 2.0 / syn_variance), axis=0)
            spectrum = (
                np.nansum(self.full_profile[nn] * data_temp / syn_variance, axis=0)
                / denom
            )
            denom_out = np.nansum(
                (self.full_profile[nn] ** 2.0 / err_temp**2.0), axis=0
            )
            errorspectrum = np.sqrt(1 / denom_out)

            # Sum up bpvalues to find interpoalted values in 2-sigma width
            qual_temp[self.full_profile[nn] / np.max(self.full_profile[nn]) < 0.02] = 0
            bpmap = np.nanmedian(qual_temp, axis=0).astype("int")

            weight_temp = 1 / signal.medfilt(errorspectrum, 11) ** 2

            # Iteratively add the spectra
            merged_optimal_spectrum[cut_slice] += weight_temp * spectrum
            merged_optimal_error_spectrum[cut_slice] += (
                weight_temp**2.0 * errorspectrum**2.0
            )
            merged_optimal_bpmap[cut_slice] += bpmap
            merged_optimal_weight_map[cut_slice] += weight_temp

            # Get aperture
            cen = self.p0[1]
            cen_idx = find_nearest(self.spataxis_list[nn], cen)
            extract_aperture = slice(
                int(np.floor(cen_idx - self.seeing_pix)),
                int(np.ceil(cen_idx + self.seeing_pix)),
            )

            # Do normal sum
            s_spectrum = np.nansum(data_temp[extract_aperture, :], axis=0)
            s_errorspectrum = np.sqrt(
                np.nansum(err_temp[extract_aperture, :] ** 2, axis=0)
            )
            s_bpmap = np.sum(qual_temp[extract_aperture, :], axis=0)

            s_weight_temp = 1 / signal.medfilt(s_errorspectrum, 11) ** 2

            # Iteratively add the spectra
            merged_standard_spectrum[cut_slice] += s_weight_temp * s_spectrum
            merged_standard_error_spectrum[cut_slice] += (
                s_weight_temp**2.0 * s_errorspectrum**2.0
            )
            merged_standard_bpmap[cut_slice] += s_bpmap
            merged_standard_weight_map[cut_slice] += s_weight_temp

        # Scale by the summed weights.
        self.optimal_flux_spec = merged_optimal_spectrum / merged_optimal_weight_map
        self.optimal_error_spec = np.sqrt(
            merged_optimal_error_spectrum / merged_optimal_weight_map**2.0
        )

        # Scale by the summed weights.
        self.standard_flux_spec = merged_standard_spectrum / merged_standard_weight_map
        self.standard_error_spec = np.sqrt(
            merged_standard_error_spectrum / merged_standard_weight_map**2.0
        )

        test_file = fits.open(
            glob.glob(
                "/Users/jonatanselsing/Work/DATA/test_data/reduced_data/OB1_NOD/UVB/*/*SLIT_FLUX_IDP*.fits"
            )[0]
        )

        wave = test_file[1].data.field("WAVE").flatten()
        flux = test_file[1].data.field("FLUX").flatten()
        error = test_file[1].data.field("ERR").flatten()
        pl.plot(wave, flux / error, label="ESO pipe")

        pl.plot(wave, self.optimal_flux_spec / self.optimal_error_spec, label="optimal")

        pl.plot(
            wave, self.standard_flux_spec / self.standard_error_spec, label="aperture"
        )

        pl.legend()
        # pl.show()

        pl.plot(wave, flux / 1e-17, label="ESO pipe")
        pl.plot(wave, self.optimal_flux_spec, label="optimal")
        pl.plot(wave, self.standard_flux_spec, label="aperture")
        pl.plot(wave, merged_standard_bpmap, linestyle="steps-mid")
        pl.legend()
        pl.show()

    def make_empty_merged_out_frame(self):
        # Get the total wavelength size in pixels
        self.nxpixel = (
            int(round((self.end_wave_list[0] - self.start_wave_list[-1]) / self.cdelt1))
            + 1
        )

        # Pick refence wcvs
        self.ref_wcs = self.wcs_outer[0][-1]

        # Construct empty output arrays - use deepcopy to make lists
        self.data_new = np.zeros((self.nypixel, self.nxpixel))
        self.err_new = copy.deepcopy(self.data_new)
        self.qual_new = copy.deepcopy(self.data_new)

    @numba.jit(parallel=True)
    def merge_orders(self):
        """
        Method to merge the orders into a single, 2D image.
        """

        weight_map = copy.deepcopy(self.data_new)
        count_map = copy.deepcopy(self.data_new)

        for nn in numba.prange(self.norders):
            # In zero based coordinates
            start_pix = int(
                np.floor(self.ref_wcs.wcs_world2pix(self.start_wave_list[nn], 0, 0)[0])
            )
            end_pix = int(
                np.floor(self.ref_wcs.wcs_world2pix(self.end_wave_list[nn], 0, 0)[0])
            )

            # Define array slice
            cut_slice = slice(start_pix, end_pix + 1)

            # Single image input
            data_temp = self.hdu[nn * 3].data
            err_temp = self.hdu[nn * 3 + 1].data
            qual_temp = self.hdu[nn * 3 + 2].data

            # Subtract sky
            # data_temp -= self.sky_background[nn]

            # Make boolean mask based on bad pixel map
            mask = qual_temp > 0

            # Use mask to construct masked arrays
            data_temp[mask] = 0
            err_temp[mask] = np.nan

            # Use a smoothed version of the noise map as weights
            err_smooth = median_filter(
                inpaint_nans(copy.deepcopy(err_temp), 5), size=10, mode="reflect"
            )
            err_temp[mask] = 0

            # Smoothed inverse varience weights
            weight_temp = 1.0 / err_smooth**2.0
            weight_temp[mask] = 0

            # Construct count images to keep track of the number of contributing pixels
            count_img = np.ones_like(data_temp)

            # Iteratively add the images
            self.data_new[:, cut_slice] += weight_temp * data_temp
            self.err_new[:, cut_slice] += weight_temp**2.0 * err_temp**2.0
            self.qual_new[:, cut_slice] += mask.astype("int")
            weight_map[:, cut_slice] += weight_temp
            count_map[:, cut_slice] += count_img

        # Scale by the summed weights.
        data_scaled = self.data_new / weight_map
        err_scaled = np.sqrt(self.err_new / weight_map**2.0)

        # Mask pixels in out map, where all input values are masked
        filter_mask = count_map == self.qual_new

        # Replace masked value with nan
        data_scaled[filter_mask] = np.nan
        err_scaled[filter_mask] = np.nan
        self.qual_new[~filter_mask] = 0
        self.qual_new[filter_mask] = 1

        # Store in merged lists
        self.data_new = list(data_scaled)
        self.err_new = list(err_scaled)
        self.qual_new = list(self.qual_new)

    def write_merged_fits(self):
        self.merge_hdu[0].data = inpaint_nans(np.array(self.data_new), 5)
        self.merge_hdu[1].data = inpaint_nans(np.array(self.err_new), 5)
        self.merge_hdu[2].data = self.qual_new

        # Update WCS
        if not self.NOD:
            self.merge_hdu[0].header["CRVAL2"] = (
                self.merge_hdu[0].header["CRVAL2"] - 2 * self.cumoff[0]
            )

        # Write fits file
        self.merge_hdu.writeto(self.base_fname_out + "_merged.fits", overwrite=True)

    def make_empty_binned_merged_frame(self):
        # Get the total wavelength size in pixels
        self.bin_merge_nxpixel = int(np.floor(self.nxpixel / self.bin_factor))

        # Pick refence wcvs
        self.ref_wcs = self.wcs_outer[0][-1]

        # Construct empty output arrays
        self.data_merged_binned_new = np.zeros((self.nypixel, self.bin_merge_nxpixel))
        self.err_merged_binned_new = copy.deepcopy(self.data_new)
        self.qual_merged_binned_new = copy.deepcopy(self.data_new)

    def bin_merged(self):
        # Make binned spectrum
        bin_flux, bin_error, bin_bpmap = bin_image(
            self.merge_hdu[0].data,
            self.merge_hdu[1].data,
            self.merge_hdu[2].data,
            self.bin_factor,
            weight=False,
        )

        # Populate empty frame
        self.data_merged_binned_new = bin_flux
        self.err_merged_binned_new = bin_error
        self.qual_merged_binned_new = bin_bpmap

    def write_binned_merged_fits(self):
        # Inpaint nan values based on sourroundings
        self.merge_hdu[0].data = inpaint_nans(np.array(self.data_merged_binned_new), 5)
        self.merge_hdu[1].data = inpaint_nans(np.array(self.err_merged_binned_new), 5)
        self.merge_hdu[2].data = self.qual_merged_binned_new

        # Update WCS
        for ii in range(len(self.merge_hdu)):
            self.merge_hdu[ii].header["CDELT1"] = (
                self.merge_hdu[ii].header["CDELT1"] * self.bin_factor
            )
            self.merge_hdu[ii].header["CD1_1"] = (
                self.merge_hdu[ii].header["CD1_1"] * self.bin_factor
            )

        # Write fits file
        self.merge_hdu.writeto(
            self.base_fname_out + "_merged_binned.fits", overwrite=True
        )

    def run_combine(
        self,
        base_fname_out,
        clean,
        NOD,
        bin_factor,
        p0,
        broken_adc,
        two_comp,
        pol_degree,
        trace_mask,
    ):
        self.base_fname_out = base_fname_out
        self.clean = clean
        self.NOD = NOD
        self.bin_factor = bin_factor
        self.p0 = p0
        self.broken_adc = broken_adc
        self.two_comp = two_comp
        self.pol_degree = pol_degree
        self.trace_mask = trace_mask

        self.make_empty_order_out_frame()
        self.make_empty_binned_order_frame()
        self.make_empty_merged_out_frame()
        self.make_empty_binned_merged_frame()

        self.merge_images()
        self.write_order_fits()

        self.bin_orders()
        self.write_binned_fits()

        # self.subtract_residual_sky()

        self.fit_order_spsf()
        self.extract_order_spectra()

        self.finetune_wavlength_solution()

        self.merge_orders()
        self.write_merged_fits()

        self.bin_merged()
        self.write_binned_merged_fits()


def main():
    object_path = "/Users/jonatanselsing/Work/DATA/test_data"
    OB = "OB1"
    arm = "UVB"
    order_files = sorted(
        glob.glob(
            object_path
            + "/reduced_data/"
            + OB
            + "/"
            + arm
            + "/*/*/*SCI_SLIT_FLUX_ORDER2D_"
            + arm
            + ".fits"
        )
    )
    merged_files = sorted(
        glob.glob(
            object_path
            + "/reduced_data/"
            + OB
            + "/"
            + arm
            + "/*/*/*SCI_SLIT_FLUX_MERGE2D_"
            + arm
            + ".fits"
        )
    )
    sky_files = sorted(
        glob.glob(
            object_path
            + "/reduced_data/"
            + OB
            + "/"
            + arm
            + "/*/*/*SKY_SLIT_MERGE1D_"
            + arm
            + ".fits"
        )
    )
    sdp_files = sorted(
        glob.glob(
            object_path
            + "/reduced_data/"
            + OB
            + "/"
            + arm
            + "/*/*SCI_SLIT_FLUX_IDP_"
            + arm
            + ".fits"
        )
    )

    # print(sky_files)
    # print(sdp_files)

    # exit()

    order_2dfiles = XSHorder_object(order_files, merged_files, sky_files, sdp_files)
    order_2dfiles.run_combine(
        "%s/%s%s" % (object_path, arm, OB),
        clean=False,
        NOD=False,
        bin_factor=50,
        p0=[1e-1, -2.5, 1.5, 0, 0],
        broken_adc=False,
        two_comp=False,
        pol_degree=[3, 2],
        trace_mask=[0],
    )


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()')
    main()
