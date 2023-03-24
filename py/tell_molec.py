#!/usr/bin/env python
# -*- coding: utf-8 -*-

import molec
import glob
import numpy as np
from astropy.io import fits


def main():
    root_dir = "/Users/jonatanselsing/Work/work_rawDATA/Bertinoro/"
    outpath = root_dir + "telluric/"

    # If true, use the telluric standard to derive telluric correction.
    tell_star = True

    arms = ["VIS", "NIR"]  # "VIS", "NIR"
    OBs = ["OB1", "OB2"]
    for kk in arms:
        for ll in OBs:
            if not tell_star:
                files = glob.glob(root_dir + "final/" + kk + ll + ".fits")
                vac_air = "vac"
            if tell_star:
                allfiles = glob.glob(
                    root_dir + "reduced_data/" + ll + "_TELL/" + kk + "/*/*"
                )
                files = [ii for ii in allfiles if "IDP" in ii]
                vac_air = "air"
            n = 1
            counter = []
            for ii in files:
                fitsfile = fits.open(ii)
                print(ii)
                sn = np.median(fitsfile[1].data.field("FLUX")) / np.median(
                    fitsfile[1].data.field("ERR")
                )

                obj_name = kk + ll + "_" + str(n)
                if obj_name in counter:
                    n += 1
                elif obj_name not in counter:
                    n = 1
                counter.append(obj_name)

                m_fit = molec.molecFit(ii, obj_name, outpath, vac_air=vac_air)
                m_fit.setParams()
                m_fit.runMolec()
                m_fit.updateSpec()


if __name__ == "__main__":
    main()
