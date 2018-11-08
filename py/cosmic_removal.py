#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import astroscrappy
import glob
from astropy.io import fits
import os

object_name = "/Users/jonatanselsing/Work/work_rawDATA/Crab_Pulsar"

for nn in glob.glob(object_name+"/data_with_raw_calibs/*cosmicced*"):
    os.remove(nn)

files = glob.glob(object_name+"/data_with_raw_calibs/*.fits")
for n in files:
    try:
        fitsfile = fits.open(str(n))
    except:
        continue
    try:
        fitsfile[0].header['HIERARCH ESO DPR CATG'] = fitsfile[0].header['HIERARCH ESO DPR CATG']
    except:
        continue

    if fitsfile[0].header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and (fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'UVB' or fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'VIS' or fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'NIR'):
        print('Removing cosmics from file: '+n+'...')

        if fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'UVB' or fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'VIS':
            gain = fitsfile[0].header['HIERARCH ESO DET OUT1 GAIN']
            ron = fitsfile[0].header['HIERARCH ESO DET OUT1 RON']
            frac = 0.01
            if fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'UVB':
                objlim = 15
                sigclip = 5
            elif fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'VIS':
                objlim = 15
                sigclip = 5
            niter = 10
        elif fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'NIR':
            gain = 2.12
            ron = 8
            frac = 0.0001
            objlim = 45
            sigclip = 20
            niter = 10

        crmask, clean_arr = astroscrappy.detect_cosmics(fitsfile[0].data, sigclip=sigclip, sigfrac=frac, objlim=objlim, cleantype='medmask', niter=niter, sepmed=True, verbose=True)

        # Replace data array with cleaned image
        fitsfile[0].data = clean_arr

        # Try to retain info of corrected pixel if extension is present.
        try:
            fitsfile[2].data[crmask] = 16 #Flag value for removed cosmic ray
        except:
            print("No bad-pixel extension present. No flag set for corrected pixels")

        # Update file
        fitsfile.writeto(n[:-5]+"cosmicced.fits", output_verify='fix')

        # Moving original file
        dirname = object_name+"/backup"
        try:
            os.mkdir(dirname)
        except:
            pass
        os.rename(n, dirname+'/'+fitsfile[0].header['ARCFILE'])
