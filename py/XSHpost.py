#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Plotting
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import seaborn; seaborn.set_style('ticks')

# Utility packages
import glob
from astropy.io import fits
from astropy import wcs
from scipy.signal import medfilt2d
from scipy.ndimage.filters import median_filter
import copy
import numpy as np
from operator import or_

from util import *


class XSHorder_object(object):
    """Object to contain a list of xshooter order 2D images"""
    def __init__(self, order_flist, merge_flist):

        # Get lists to contain
        self.filelist = order_flist
        self.merge_filelist = merge_flist

        self.nfiles = len(order_flist)
        self.hdul = [fits.open(ii) for ii in self.filelist]
        self.merge_hdul = [fits.open(ii) for ii in self.merge_filelist]

        # Make reference hdus
        self.hdu = self.hdul[0]
        self.merge_hdu = self.merge_hdul[0]

        # Get order header information
        self.get_order_headers()

        # Get order wcs
        self.get_order_wcs()

        # Get arm
        self.arm = self.merge_hdu[0].header['HIERARCH ESO SEQ ARM']


    def get_order_headers(self):

        # Get number of orders in image
        norders = [int(len(ii) / 3) for ii in self.hdul]

        # Images must have same number of orders
        if not norders.count(norders[0]) == len(norders):
            raise TypeError("Input order list does not have the same number of orders.")

        self.norders = norders[0]

        # Make a nested list with all order headers
        self.order_header_outer = [[kk[ll*3].header for ll in range(self.norders)] for kk in self.hdul]

        # Construct transposed list to loop through spatial dir
        self.order_header_outer_t = list(map(list, zip(*self.order_header_outer)))

    def get_order_wcs(self):

        # Construct wcs, using each order header for all files
        self.wcs_outer = [[wcs.WCS(header) for header in hdus] for hdus in self.order_header_outer]

        # Construct transposed list to loop through spatial dir
        self.wcs_outer_t = list(map(list, zip(*self.wcs_outer)))

        # Find number of order elements
        self.naxis1_list = [header['NAXIS1'] for header in self.order_header_outer[0]]
        self.naxis2_list = [header['NAXIS2'] for header in self.order_header_outer[0]]

        # Images must have same number of spatial pixels
        if not self.naxis2_list.count(self.naxis2_list[0]) == len(self.naxis2_list):
            raise TypeError("Input image list does not have the same number of spatial pixels.")

        self.naxis2 = self.naxis2_list[0]

        self.cdelt1_list = [header['CDELT1'] for header in self.order_header_outer[0]]
        self.cdelt2_list = [header['CDELT2'] for header in self.order_header_outer[0]]

        # Images must have same sampling in both directions
        if not self.cdelt1_list.count(self.cdelt1_list[0]) == len(self.cdelt1_list):
            raise TypeError("Input image list does not have the same wavelength sampling.")
        if not self.cdelt2_list.count(self.cdelt2_list[0]) == len(self.cdelt2_list):
            raise TypeError("Input image list does not have the same spatial sampling.")

        self.cdelt1 = self.cdelt1_list[0]
        self.cdelt2 = self.cdelt2_list[0]

        # Get wavelength extent of orders in nm
        self.start_wave_list = [wcs.wcs_pix2world(1, 1, 1)[0] + 0 for wcs in self.wcs_outer[0]]
        self.end_wave_list = [wcs.wcs_pix2world(naxis1, 1, 1)[0] + 0 for wcs, naxis1 in zip(self.wcs_outer[0], self.naxis1_list)]

        # Spatial offset of individual images in arcsec
        self.cumoff = [header['HIERARCH ESO SEQ CUMOFF Y'] for header in self.order_header_outer_t[0]]

        # Get spatial extent of orders by looping over the transposed list
        self.start_size_list = [wcs.wcs_pix2world(1, 1, 1)[1] + self.cumoff[elem] for elem, wcs in enumerate(self.wcs_outer_t[0])]
        self.end_size_list = [wcs.wcs_pix2world(1, naxis2, 1)[1] + self.cumoff[elem] for elem, (wcs, naxis2) in enumerate(zip(self.wcs_outer_t[0], self.naxis2_list))]


    def make_empty_order_out_frame(self):

        # Get spatial offsets for total output image.
        max_neg_offset = min(self.start_size_list + self.end_size_list)
        max_pos_offset = max(self.start_size_list + self.end_size_list)

        # Get pixel size of output image
        self.nypixel = int(round((max_pos_offset - max_neg_offset) / self.cdelt2)) + 1

        # Pick refence wcvs
        self.ref_wcs = self.wcs_outer[0][-1]

        # Construct empty output arrays
        self.data_order_new = [np.zeros((self.nypixel, self.naxis1_list[ii])) for ii, order in enumerate(self.order_header_outer[0])]
        self.err_order_new = copy.deepcopy(self.data_order_new)
        self.qual_order_new = copy.deepcopy(self.data_order_new)


    def make_clean_mask(self):

        # Mask outliers in the direction of the stack and "bad" pixels.

        # Loop through orders
        for kk in range(self.norders):

            # Make storage cubes
            shp = np.shape(self.data_order_new[kk])
            data_store_cube = np.zeros((shp[0], shp[1], self.nfiles))
            err_store_cube = np.zeros((shp[0], shp[1], self.nfiles))
            qual_store_cube = np.zeros((shp[0], shp[1], self.nfiles))

            # Populate cube
            for ii in range(self.nfiles):
                # In zero based coordinates
                start_pix = int(round(np.floor(self.ref_wcs.wcs_world2pix(0 ,self.start_size_list[ii], 0)[1])))
                end_pix = int(round(np.floor(self.ref_wcs.wcs_world2pix(0, self.end_size_list[ii], 0)[1])))

                # Get offset to start at pixel 0
                min_pix = abs(min([start_pix, end_pix]))

                # Define array slice
                cut_slice = slice(start_pix + min_pix, end_pix + min_pix + 1)

                # Single image input
                data_temp = self.hdul[ii][kk*3].data/1e-17
                err_temp = self.hdul[ii][kk*3 + 1].data/1e-17
                qual_temp = self.hdul[ii][kk*3 + 2].data

                data_store_cube[cut_slice, :, ii] = data_temp
                err_store_cube[cut_slice, :, ii] = err_temp
                qual_store_cube[cut_slice, :, ii] = qual_temp


            # Mask 3(5)-sigma outliers in the direction of the stack
            m, s = np.ma.median(np.ma.array(data_store_cube, mask=qual_store_cube), axis = 2).data,  np.std(np.ma.array(data_store_cube, mask=qual_store_cube), axis = 2).data
            if self.arm == "NIR":
                sigma_mask = 5
            else:
                sigma_mask = 3
            l, h = np.tile((m - sigma_mask*s).T, (self.nfiles, 1, 1)).T, np.tile((m + sigma_mask*s).T, (self.nfiles, 1, 1)).T
            qual_store_cube[(data_store_cube < l) | (data_store_cube > h)] = 666


            # Introduce filter based on smoothed image
            flux_filt, error_filt, bp_filt = np.zeros_like(data_store_cube), np.zeros_like(data_store_cube), np.zeros_like(data_store_cube)
            for ii in range(data_store_cube.shape[2]):
                flux_filt[:, :, ii] = medfilt2d(data_store_cube[:, :, ii], 3)
                error_filt[:, :, ii] = medfilt2d(err_store_cube[:, :, ii], 3)

            bp_filt = abs(flux_filt - data_store_cube) > 7 * error_filt
            bp_filt[int(np.floor(self.nypixel/2-2)):int(np.ceil(self.nypixel/2+2)), :, :] = False
            qual_store_cube[bp_filt == True] = 555

            # Probagate the updated bad pixel maps to the individual orders
            for ii in range(self.nfiles):

                # In zero based coordinates
                start_pix = int(round(np.floor(self.ref_wcs.wcs_world2pix(0 ,self.start_size_list[ii], 0)[1])))
                end_pix = int(round(np.floor(self.ref_wcs.wcs_world2pix(0, self.end_size_list[ii], 0)[1])))

                # Get offset to start at pixel 0
                min_pix = abs(min([start_pix, end_pix]))

                # Define array slice
                cut_slice = slice(start_pix + min_pix, end_pix + min_pix + 1)

                # Make mask based on bpmaps
                mask = (qual_store_cube > 0).astype("int")
                self.hdul[ii][kk*3 + 2].data = mask[cut_slice, :, ii]





    def merge_images(self):

        weight_map = copy.deepcopy(self.data_order_new)
        count_map = copy.deepcopy(self.data_order_new)

        for kk in range(self.norders):

            if self.clean:
                self.make_clean_mask()

            if self.NOD:
                self.data_order_new[kk] = self.hdul[0][kk*3].data/1e-17
                self.err_order_new[kk] = self.hdul[0][kk*3 + 1].data/1e-17
                self.qual_order_new[kk] = self.hdul[0][kk*3 + 2].data
                continue


            for ii in range(self.nfiles):

                # In zero based coordinates
                start_pix = int(round(np.floor(self.ref_wcs.wcs_world2pix(0 ,self.start_size_list[ii], 0)[1])))
                end_pix = int(round(np.floor(self.ref_wcs.wcs_world2pix(0, self.end_size_list[ii], 0)[1])))

                # Get offset to start at pixel 0
                min_pix = abs(min([start_pix, end_pix]))

                # Define array slice
                cut_slice = slice(start_pix + min_pix, end_pix + min_pix + 1)

                # Single image input
                data_temp = self.hdul[ii][kk*3].data/1e-17
                err_temp = self.hdul[ii][kk*3 + 1].data/1e-17
                qual_temp = self.hdul[ii][kk*3 + 2].data

                # Make boolean mask based on bad pixel map
                mask = (qual_temp > 0)

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
                    err_smooth = median_filter(inpaint_nans(copy.deepcopy(err_temp), 5), size = 10, mode="reflect")

                err_temp[mask] = 0

                # Smoothed inverse varience weights
                weight_temp = 1./err_smooth**2.
                weight_temp[mask] = 0

                # Construct count images to keep track of the number of contributing pixels
                count_img = np.ones_like(data_temp)

                # Iteratively add the images
                self.data_order_new[kk][cut_slice, :] += weight_temp * data_temp
                self.err_order_new[kk][cut_slice, :] += weight_temp**2. * err_temp**2.
                self.qual_order_new[kk][cut_slice, :] += mask.astype("int")
                weight_map[kk][cut_slice, :] += weight_temp
                count_map[kk][cut_slice, :] += count_img

            # Scale by the summed weights.
            data_scaled = self.data_order_new[kk] / weight_map[kk]
            err_scaled = np.sqrt(self.err_order_new[kk] / weight_map[kk]**2.)

            # Mask pixels in out map, where all input values are masked
            filter_mask = (count_map[kk] == self.qual_order_new[kk])

            # Replace masked value with nan
            data_scaled[filter_mask] = np.nan
            err_scaled[filter_mask] = np.nan
            self.qual_order_new[kk][~filter_mask] = 0
            self.qual_order_new[kk][filter_mask] = 1

            # Store in order lists
            self.data_order_new[kk] = list(data_scaled)
            self.err_order_new[kk] = list(err_scaled)
            self.qual_order_new[kk] = list(self.qual_order_new[kk])



    def write_order_fits(self):
        """
        Method to save combined image
        """

        for ii, kk in enumerate(self.hdu):
            count = int(np.floor(ii/3))
            self.hdu[count*3].data = self.data_order_new[count]
            self.hdu[count*3 + 1].data = self.err_order_new[count]
            self.hdu[count*3 + 2].data = self.qual_order_new[count]

            # Update WCS
            self.hdu[ii].header["CRVAL2"] = self.hdu[ii].header["CRVAL2"] - 2 * self.cumoff[0]

        # Write file
        self.hdu.writeto(self.base_fname_out+"_orders.fits", overwrite=True)



    def make_empty_binned_order_frame(self):

        # Get the total wavelength size in pixels
        self.bin_nxpixel = (np.around((np.array(self.end_wave_list) - np.array(self.start_wave_list))/ (self.cdelt1*self.bin_factor))).astype("int") + 1

        # Pick refence wcvs
        self.ref_wcs = self.wcs_outer[0][-1]

        # Construct empty output arrays
        self.data_order_binned_new = [np.zeros((self.naxis2, self.bin_nxpixel[ii])) for ii, order in enumerate(self.order_header_outer[0])]
        self.err_order_binned_new = copy.deepcopy(self.data_order_binned_new)
        self.qual_order_binned_new = copy.deepcopy(self.data_order_binned_new)



    def bin_orders(self):
        # Loop through the orders
        for ii in range(self.norders):

            # Make binned spectrum
            bin_flux, bin_error, bin_bpmap = bin_image(self.hdu[ii*3].data, self.hdu[ii*3 + 1].data, self.hdu[ii*3 + 2].data, self.bin_factor, weight = False)

            # Populate empty frame
            self.data_order_binned_new[ii] = bin_flux
            self.err_order_binned_new[ii] = bin_error
            self.qual_order_binned_new[ii] = bin_bpmap



    def write_binned_fits(self):

        for ii, kk in enumerate(self.hdu):
            count = int(np.floor(ii/3))
            self.hdu[count*3].data = self.data_order_binned_new[count]
            self.hdu[count*3 + 1].data = self.err_order_binned_new[count]
            self.hdu[count*3 + 2].data = self.qual_order_binned_new[count]

            # Update WCS
            self.hdu[ii].header["CDELT1"] = self.hdu[ii].header["CDELT1"]*self.bin_factor

        # Write file
        self.hdu.writeto(self.base_fname_out+"_orders_binned.fits", overwrite=True)

        for ii, kk in enumerate(self.hdu):
            count = int(np.floor(ii/3))
            self.hdu[count*3].data = self.data_order_new[count]
            self.hdu[count*3 + 1].data = self.err_order_new[count]
            self.hdu[count*3 + 2].data = self.qual_order_new[count]

            # Update WCS
            self.hdu[ii].header["CDELT1"] = self.hdu[ii].header["CDELT1"]/self.bin_factor





    # def fit_spsf(self):



    def make_empty_merged_out_frame(self):

        # Get the total wavelength size in pixels
        self.nxpixel = int(round((self.end_wave_list[0] - self.start_wave_list[-1]) / self.cdelt1)) + 1

        # Pick refence wcvs
        self.ref_wcs = self.wcs_outer[0][-1]

        # Construct empty output arrays - use deepcopy to make lists
        self.data_new = np.zeros((self.nypixel, self.nxpixel))
        self.err_new = copy.deepcopy(self.data_new)
        self.qual_new = copy.deepcopy(self.data_new)


    def merge_orders(self):
        """
        Method to merge the orders into a single, 2D image.
        """

        weight_map = copy.deepcopy(self.data_new)
        count_map = copy.deepcopy(self.data_new)

        for ii in range(self.norders):
            # In zero based coordinates
            start_pix = int(np.floor(self.ref_wcs.wcs_world2pix(self.start_wave_list[ii],0 ,0)[0]))
            end_pix = int(np.floor(self.ref_wcs.wcs_world2pix(self.end_wave_list[ii],0 ,0)[0]))

            # Define array slice
            cut_slice = slice(start_pix, end_pix+1)

            # Single image input
            data_temp = self.hdu[ii*3].data
            err_temp = self.hdu[ii*3 + 1].data
            qual_temp = self.hdu[ii*3 + 2].data

            # Make boolean mask based on bad pixel map
            mask = (qual_temp > 0)

            # Use mask to construct masked arrays
            data_temp[mask] = 0
            err_temp[mask] = np.nan

            # Use a smoothed version of the noise map as weights
            err_smooth = median_filter(inpaint_nans(copy.deepcopy(err_temp), 5), size = 10, mode="reflect")
            err_temp[mask] = 0

            # Smoothed inverse varience weights
            weight_temp = 1./err_smooth**2.
            weight_temp[mask] = 0

            # Construct count images to keep track of the number of contributing pixels
            count_img = np.ones_like(data_temp)

            # Iteratively add the images
            self.data_new[:, cut_slice] += weight_temp * data_temp
            self.err_new[:, cut_slice] += weight_temp**2. * err_temp**2.
            self.qual_new[:, cut_slice] += mask.astype("int")
            weight_map[:, cut_slice] += weight_temp
            count_map[:, cut_slice] += count_img

        # Scale by the summed weights.
        data_scaled = self.data_new / weight_map
        err_scaled = np.sqrt(self.err_new / weight_map**2.)

        # Mask pixels in out map, where all input values are masked
        filter_mask = (count_map == self.qual_new)

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
            self.merge_hdu[0].header["CRVAL2"] = self.merge_hdu[0].header["CRVAL2"] - 2 * self.cumoff[0]

        # Write fits file
        self.merge_hdu.writeto(self.base_fname_out+"_merged.fits", overwrite=True)


    def make_empty_binned_merged_frame(self):

        # Get the total wavelength size in pixels
        self.bin_nxpixel = (np.around((np.array(self.end_wave_list) - np.array(self.start_wave_list))/ (self.cdelt1*self.bin_factor))).astype("int") + 1

        # Pick refence wcvs
        self.ref_wcs = self.wcs_outer[0][-1]

        # Construct empty output arrays
        self.data_merged_binned_new = [np.zeros((self.naxis2, self.bin_nxpixel[ii])) for ii, order in enumerate(self.order_header_outer[0])]
        self.err_merged_binned_new = copy.deepcopy(self.data_merged_binned_new)
        self.qual_merged_binned_new = copy.deepcopy(self.data_merged_binned_new)



    def bin_merged(self):
        # Loop through the orders
        for ii in range(self.norders):

            # Make binned spectrum
            bin_flux, bin_error, bin_bpmap = bin_image(self.hdu[ii*3].data, self.hdu[ii*3 + 1].data, self.hdu[ii*3 + 2].data, self.bin_factor, weight = False)

            # Populate empty frame
            self.data_merged_binned_new[ii] = bin_flux
            self.err_merged_binned_new[ii] = bin_error
            self.qual_merged_binned_new[ii] = bin_bpmap



    def write_binned_merged_fits(self):


        self.merge_hdu[0].data = inpaint_nans(np.array(self.data_merged_binned_new), 5)
        self.merge_hdu[1].data = inpaint_nans(np.array(self.err_merged_binned_new), 5)
        self.merge_hdu[2].data = self.qual_merged_binned_new

        # Update WCS
        if not self.NOD:
            self.merge_hdu[0].header["CRVAL2"] = self.merge_hdu[0].header["CRVAL2"] - 2 * self.cumoff[0]

        # Write fits file
        self.merge_hdu.writeto(self.base_fname_out+"_merged_binned.fits", overwrite=True)






    def run_combine(self, base_fname_out, clean, NOD, bin_factor):
        self.base_fname_out = base_fname_out
        self.clean = clean
        self.NOD = NOD
        self.bin_factor = bin_factor

        self.make_empty_order_out_frame()
        self.merge_images()
        self.write_order_fits()

        self.make_empty_binned_order_frame()
        self.bin_orders()
        self.write_binned_fits()

        # self.fit_spsf()


        self.make_empty_merged_out_frame()
        self.merge_orders()
        self.write_merged_fits()

        self.make_empty_binned_merged_frame()
        self.bin_merged()
        self.write_binned_merged_fits()


def main():

    object_path = "/Users/jonatanselsing/Work/work_rawDATA/Bertinoro"
    OB = "OB1_NOD"
    arm = "VIS"
    order_files = sorted(glob.glob(object_path+"/reduced_data/"+OB+"/"+arm+"/*/*/*SCI_SLIT_FLUX_ORDER2D_"+arm+".fits"))
    merged_files = sorted(glob.glob(object_path+"/reduced_data/"+OB+"/"+arm+"/*/*/*SCI_SLIT_FLUX_MERGE2D_"+arm+".fits"))
    # order_sky_files = sorted(glob.glob(object_path+"/rbin_nxpixeleduced_data/"+OB+"/"+arm+"/*/*/SCI_SLIT_SKY_ORD1D_"+arm+".fits"))

    order_2dfiles = XSHorder_object(order_files, merged_files)
    order_2dfiles.run_combine("%s/%s%s"%(object_path, arm, OB), clean=False, NOD=True, bin_factor = 10)







if __name__ == '__main__':
    # import cProfile
    # cProfile.run('main()')
    main()








    # Disable nodding for now. The sky is subtracted in the unrectified image, which means that we cannot access the order sky
    # def form_nodding_pairs(self):

    #     # Loop through orders
    #     for kk in range(self.norders):

    #         # Make storage cubes
    #         shp = np.shape(self.data_order_new[kk])
    #         data_store_cube = np.zeros((shp[0], shp[1], self.nfiles))
    #         err_store_cube = np.zeros((shp[0], shp[1], self.nfiles))
    #         qual_store_cube = np.zeros((shp[0], shp[1], self.nfiles))
    #         pix_offset_y = np.zeros(self.nfiles)

    #         # Populate cube
    #         for ii in range(self.nfiles):
    #             # In zero based coordinates
    #             start_pix = int(round(np.floor(self.ref_wcs.wcs_world2pix(0 ,self.start_size_list[ii], 0)[1])))
    #             end_pix = int(round(np.floor(self.ref_wcs.wcs_world2pix(0, self.end_size_list[ii], 0)[1])))

    #             # Get offset to start at pixel 0
    #             min_pix = abs(min([start_pix, end_pix]))

    #             # Define array slice
    #             cut_slice = slice(start_pix + min_pix, end_pix + min_pix + 1)

    #             # Single image input
    #             data_temp = self.hdul[ii][kk*3].data
    #             err_temp = self.hdul[ii][kk*3 + 1].data
    #             qual_temp = self.hdul[ii][kk*3 + 2].data

    #             data_store_cube[cut_slice, :, ii] = data_temp
    #             err_store_cube[cut_slice, :, ii] = err_temp
    #             qual_store_cube[cut_slice, :, ii] = qual_temp

    #             pix_offset_y[ii] = start_pix + min_pix

    #         # Form nodding pairs
    #         if self.NOD:
    #             if not self.nirrepeats == 1:
    #                 # Smaller container
    #                 flux_cube_tmp = np.zeros((self.nxpixel, self.nypixel, int(np.ceil(self.nfiles / self.nirrepeats))))
    #                 error_cube_tmp = np.zeros((self.nxpixel, self.nypixel, int(np.ceil(self.nfiles / self.nirrepeats))))
    #                 bpmap_cube_tmp = np.zeros((self.nxpixel, self.nypixel, int(np.ceil(self.nfiles / self.nirrepeats))))

    #                 # Collapse in self.nirrepeats
    #                 for ll, pp in enumerate(np.arange(int(np.ceil(self.nfiles / self.nirrepeats)))):
    #                     # Make lower an upper index of files, which is averaged over. If all NOD positions does not have the same number of self.nirrepeats, assume the last position is cut.
    #                     low, up = ll*self.nirrepeats, min(self.nfiles, (ll+1)*self.nirrepeats)
    #                     # Slice structure
    #                     subset = slice(low, up)
    #                     # Average over subset
    #                     flux_cube_tmp[:, :, ll], error_cube_tmp[:, :, ll], bpmap_cube_tmp[:, :, ll] = avg(data_store_cube[:, :, subset], err_store_cube[:, :, subset], qual_store_cube[:, :, subset].astype("bool"), axis=2)
    #                 # Update number holders
    #                 self.nfiles_list = np.arange(self.nfiles/self.nirrepeats)
    #                 # pix_offsety = pix_offsety[::self.nirrepeats]
    #                 data_store_cube, err_store_cube, qual_store_cube = flux_cube_tmp, error_cube_tmp, bpmap_cube_tmp

    #             # Form the pairs [(A1-B1) - shifted(B1-A1)] and [(B2-A2) - shifted(A2-B2)] at positions 0, 2. Sets the other images to np.nan.
    #             data_store_cube, err_store_cube, qual_store_cube = form_nodding_pairs(data_store_cube, err_store_cube,  qual_store_cube, self.naxis2, pix_offset_y.astype("int"))



    #         # Probagate the nodded frames to the individual orders
    #         for ii in range(self.nfiles):
    #             # In zero based coordinates
    #             start_pix = int(round(np.floor(self.ref_wcs.wcs_world2pix(0 ,self.start_size_list[ii], 0)[1])))
    #             end_pix = int(round(np.floor(self.ref_wcs.wcs_world2pix(0, self.end_size_list[ii], 0)[1])))

    #             # Get offset to start at pixel 0
    #             min_pix = abs(min([start_pix, end_pix]))

    #             # Define array slice
    #             cut_slice = slice(start_pix + min_pix, end_pix + min_pix + 1)

    #             # Make mask based on bpmaps
    #             self.hdul[ii][kk*3].data = data_store_cube[cut_slice, :, ii]
    #             self.hdul[ii][kk*3 + 1].data = err_store_cube[cut_slice, :, ii]
    #             self.hdul[ii][kk*3 + 2].data = qual_store_cube[cut_slice, :, ii]


    # def put_back_sky(self):
    #     # Loop through orders
    #     for kk in range(self.norders):
    #         # Loop though files
    #         for ii in range(self.nfiles):
    #             # Put back the flux-calibrated sky
    #             pl.imshow(self.hdul_sky[ii][kk*3].data/self.hdul_resp[ii][kk].data)
    #             pl.show()
    #             self.hdul[ii][kk*3].data += self.hdul_sky[ii][kk*3].data/self.hdul_resp[ii][kk].data


