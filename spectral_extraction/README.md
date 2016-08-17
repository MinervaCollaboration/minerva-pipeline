# MINERVA Raw Reduction/Spectral Extraction Documentation (Optimal Extraction Version)

**Purpose:**

Convert ccd frames to extracted spectra with wavelength solution (not currently normalized with flats or flux calibration)

**Inputs:**

Calibration Data
- Arc frames (two per telescope, one with, one without iodine)
- Fiber flats (one per telescope)
- Slit flats (not presently taken)

Science Data (All the same from point of view of code)
- Daytime Sky
- B-Stars
- Target Stars

**Outputs:**

Extracted Spectrum - .fits file

Four HDU extensions:

1. Photon Counts (proxy for flux)
2. Wavelength
3. Inverse Variance
4. Pixel Mask

Each extension contains a 3D array:

1. Axis 1 = pixel position (along trace)
2. Axis 2 = order
3. Axis 3 = telescope
 
**_View details for installation, operation, and an explanation of the extraction procedure in docs/MINERVA Extraction Documentation_**
