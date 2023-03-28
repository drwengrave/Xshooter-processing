# X-shooter post processing scripts
Scripts created for post-processing [X-shooter](https://www.eso.org/sci/facilities/paranal/instruments/xshooter.html) reductions.
In particular, the scripts provided here allow to combine data taken in NODDING mode but reduced in STARE mode, as well as extract a 1D spectrum from a 2D spectrum.


I have forked this repository from [Jonatan Selsing](https://github.com/jselsing) and tidied things up for ease of use but all credit goes to him.


## Installation

To use the scripts, clone and download this repository (or download the .zip). 
```
git clone https://github.com/JPalmerio/xsh-postproc.git
```

There is a up-to-date `conda` environment file in order to install the proper python environment, which can be done with:
```
conda env create -f xsh_postproc_env.yaml
```
This will create a `conda` environment named `xsh_postproc` which you can activate with:

```
conda activate xsh_postproc
```


## Usage

The two main scripts in the package are `XSHcomb.py` and `XSHextract.py.`
These take care of combinations of individual exposures and 1D-extractions respectively.
The idea is that the ESO X-shooter pipeline is used to reduce all observations in STARE-mode, and then the scripts provided here, do combinations and extractions where the X-shooter pipeline can be improved. 

### Prerequisites

Install the ESO pipelines using the installation instructions available at https://www.eso.org/sci/software/pipelines/.
The file `xsh_workflow_stare.kar` contains a workflow which can be used with the X-Shooter pipeline, which has been preconfigured to reduce the observations in STARE-mode.

> :warning: Don't forget to set the `RAW_DATA_DIR` to the folder containing your **_unzipped_** data and `ROOT_DATA_DIR` to where you want the reduced dataset to be saved.

When esoreflex is installed and the workflow is loaded into Kepler (ESO workflow engine), it should look something like this:

![alt tag](docs/figs/esoreflex.png)


During processing of the workflow, two quality-control windows are shown:

- **The flux-standard reduction**, where the response function is computed.
This window should be inspected for agreement between the blue and the green lines, signifying that the standard star has been adequately calibrated.
The blue line is the nightly, flux-calibrated standard star and the green line tabulated flux for this star. 

- **The science object reduction**, where mainly the sky regions should be set for each element in the nodding sequence.
It could look something like this, where a faint trace of the afterglow is visible, centered at -2.5 arcsec.
The sky is specified using the sky_position and sky-hheight. For this example, two sky windows have been chosen, one at 2 arcsecond with a half-height of 3 arcseconds, and one at -5 arcsec with a 1 arcsec half-height.

![alt tag](docs/figs/sky_sub.png)


Each complete nodding sequence will produce 4 individual reductions (A,B,B,A) for each arm.
The scripts provided in this repository aim to combine these individual reductions into one final product.
After running `esoreflex` and before moving any files from the output, run the following script:
```
python xshpp_sort_esoreflex_stare_output.py ESOREFLEX_ROOT_DATA_DIR
```
where `ESOREFLEX_ROOT_DATA_DIR` is the `ROOT_DATA_DIR` you provided in the `ESOREFLEX` workflow.
> :bulb: **Tip:** Try to reduce all products in one go. 
> The reason for this is that each time you make a new reduction, `ESOREFLEX` creates a new output directory for the end products, named after the time that you started the reduction.
> But it does not create the same directory for the temporary files which are needed to run the post-processing script!
> This means associating the temporary files with their correct reduction is not trivial.


![alt tag](docs/figs/XSHcomb.png)

example usage

$
python XSHcomb.py /Users/jonatanselsing/Work/work_rawDATA/Crab_Pulsar/ UVB STARE OB9
$

and 

![alt tag](docs/figs/XSHextract.png)

$
python XSHextract.py /Users/jonatanselsing/Work/work_rawDATA/Crab_Pulsar/UVBOB9skysub.fits  --optimal 
$

## License
-------

Copyright 2016-2020 Jonatan Selsing and contributors.

These scripts are free software made available under the GNU License. For details see
the LICENSE file.

