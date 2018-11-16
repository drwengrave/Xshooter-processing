#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

from astropy.io import fits
import numpy as np
import glob


def main():

    root_dir = "/Users/jonatanselsing/Work/work_rawDATA/Crab_Pulsar/"

    arms = ["UVB"]  # ["UVB", "VIS", "NIR"]
    # OBs = ["OB1", "OB2", "OB3", "OB4", "OB5", "OB6", "OB7", "OB8", "OB9", "OB10", "OB11", "OB12", "OB13"]
    OBs = ["OB9"]

    ext_name = "skysubstdext.dat"  # None

    tell_file = 1

    for ii, arm in enumerate(arms):
        for kk, OB in enumerate(OBs):

            print(root_dir+arm+OB)

            # if OBs is None:
            dat = np.genfromtxt(root_dir+arm+OB+ext_name)

            try:
                wl, f, e, bpmap, dust, resp, slitcorr = dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4], dat[:, 5], dat[:, 6], dat[:, 7]
            except:
                wl, f, e, bpmap, dust, slitcorr = dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4], dat[:, 5], dat[:, 6]


            file = glob.glob(root_dir + "reduced_data/"+OB+"/"+arm+"/*/*_IDP_"+arm+".fits")[0]
            n_files = len(glob.glob(root_dir + "reduced_data/"+OB+"/"+arm+"/*/*_IDP_"+arm+".fits"))

            print(file)

            fitsfile = fits.open(file)

            # Read in telluric correction
            print(root_dir + "telluric/" + arm + OB + "TELL"+str(tell_file)+"_TAC.fits")
            try:
                t = t_file[1].data.field("mtrans").flatten()
            except:
                t = np.ones_like(wl)
            print(t)
            # Update data columns
            c = fitsfile[1].columns["WAVE"]
            c.data = (wl/10) #* (1 - fitsfile[0].header['HIERARCH ESO QC VRAD BARYCOR']/3e5)
            fitsfile[1].data["WAVE"] = c.data

            c = fitsfile[1].columns["FLUX"]
            c.data = f*slitcorr
            fitsfile[1].data["FLUX"] = c.data

            c = fitsfile[1].columns["ERR"]
            c.data = e*slitcorr
            fitsfile[1].data["ERR"] = c.data

            c = fitsfile[1].columns["QUAL"]
            c.data = bpmap
            fitsfile[1].data["QUAL"] = c.data

            c = fitsfile[1].columns["SNR"]
            c.data = t
            c.name = "TRANS"
            fitsfile[1].data["TRANS"] = c.data

            try:
                c = fitsfile[1].columns["FLUX_REDUCED"]
                c.data = f/resp
                fitsfile[1].data["FLUX_REDUCED"] = c.data
                c = fitsfile[1].columns["ERR_REDUCED"]
                c.data = e/resp
                fitsfile[1].data["ERR_REDUCED"] = c.data
            except:
                pass

            # Update header values
            fitsfile[1].header["TELAPSE"] = fitsfile[1].header["TELAPSE"] * n_files
            fitsfile[0].header["EXPTIME"] = fitsfile[0].header["EXPTIME"] * n_files
            fitsfile[0].header["TEXPTIME"] = fitsfile[0].header["TEXPTIME"] * n_files

            fitsfile.writeto(root_dir+"final/"+arm+OB+".fits", overwrite=True)

            # if arm == "UVB" or arm == "VIS":
            #     fitsfile.writeto(root_dir+"final/"+arm+OB+".fits", overwrite=True)
            # elif arm == "NIR":
            #     fitsfile.writeto(root_dir+"final/"+arm+OB+".fits", overwrite=True)


if __name__ == '__main__':
    main()
