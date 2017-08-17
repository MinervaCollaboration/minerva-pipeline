## Included Code:

- **SP_extract.py** – Code for testing 2D PSF and extraction
- **arc_calibrate.py** – Finds the wavelength solution from the arc frames
- **bsplines.py** – Generates functions related to splines in support of 2D PSF
- **dark_flat_avg.py** – Meant to find mean dark, bias, frames to feed other functions
- **find_2D_psf.py** – Mast code for fitting 2D point spread function, split to parallelize.  Calls heavily on psf_utils.
- **fit_slit.py** – Imperfect code for flat fielding with slit flats.  Not currently in use
- **minerva_utils.py** – MINERVA specific functions, includes trace fitting and optimal extraction code.
- **minervasim.py** – Builds a model minerva ccd from kiwispec data, outdated, doesn't perfectly match actual spectrograph
- **modify_fits.py** – Resaves fits files from some of the earliest files that were in a weird format.  Not needed for recent data.
- **optimal_extract.py** – Code to run optimal extraction.  Calls heavily on functions in special and minerva_utils
- **psf_utils.py** – Special functions for fitting 2D PSF
- **simple_opt_ex.py** – Old code to perform optimal extraction (still in use, but will be deprecated)
- **special.py** – general functions (Gaussian profile, chi^2 fitting, etc.)
