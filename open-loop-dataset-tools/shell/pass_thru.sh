#!/bin/sh

echo This file is not for execution!
echo Exiting...
sleep 0.5
echo 3
sleep 0.5
echo 2
sleep 0.5
echo 1
exit


# Leaving core file call and all that jazz so atom colour-codes text

# Pass-thru
../../bin/viewer \
  -use_camera_model \
  \
  -detector 0 1 width height 0 \
  -irradiance 1.0 1.0 1.0 \
  -noinverse_square_law \
  -aperture 0 ignored 1 0 \
  -pixel_angle 0 unit \
  -distortion 0 0.0 0.0 0.0 0.0 0.0 \
  -tangential 0 0.0 0.0 0.0 0.0 \
  -scattering 0 ignored 0.0 0.0 0.0 0 \
  -psf_gauss 0 0 0.0 0 0.0 0 0.0 \
  -exposure 0 1 300 \
  -readout_basic 0 0 1.0 0.0 1.0 1e6 0.0 0.0 \
  -smear 0 none down 1.0 0.0 0.0 0.0 \
  -photon_noise 0 0 \
  -dark_current 0 none 0 0 \
  -detector_radiation 0 0 \
	\
	$*

# HAYABUSA Amica camera
  ../../bin/viewer \
    -use_camera_model \
    \
  	-detector 0 1 1024 1024 12 \
  	-irradiance 896099586883.56 896099586883.56 896099586883.56 \
  	-inverse_square_law \
  	-aperture 0 ignored 0.015 0 \
  	-pixel_angle 0 unit \
  	-distortion 0 0 -2.8e-5 0 0 0 \
    -tangential 0 0.0 0.0 0.0 0.0 \
    -scattering 0 ignored 0.0 0.0 0.0 0 \
  	-psf_gauss 0 16 0.005 16 0.005 0 0 \
  	-exposure 0 0.087 246.71 \
  	-readout_basic 0 0 0.02 0 17 70000 60 0 \
    -smear 0 none down 1.0 0.0 0.0 0.0 \
  	-photon_noise 0 1 \
  	-dark_current 0 none 0 0 \
  	-detector_radiation 0 0 \
  	\
  	$*

# DVS
  ../../bin/viewer \
    -use_camera_model \
    \
    -detector 0 1 128 128 0 \
    -irradiance 896099586883.56 896099586883.56 896099586883.56 \
    -inverse_square_law \
    -aperture 0 ignored 1 0 \
    -pixel_angle 0 unit \
    -distortion 0 0.0 0.0 0.0 0.0 0.0 \
    -tangential 0 0.0 0.0 0.0 0.0 \
    -scattering 0 ignored 0.0 0.0 0.0 0 \
    -psf_gauss 0 0 0.0 0 0.0 0 0.0 \
    -exposure 0 0.000015 300 \
    -readout_basic 0 0 1.0 0.0 1.0 1e6 55 24966 \
    -smear 0 none down 1.0 0.0 0.0 0.0 \
    -photon_noise 0 1 \
    -dark_current 0 constant 0 1 \
    -detector_radiation 0 0 \
  	\
  	$*

  # DVS notes:
  # -detector 0 1 128 128 0 \
  # # resolution is 128x128

  # -irradiance 896099586883.56 896099586883.56 896099586883.56 \
  # -inverse_square_law \
  # # just took it from the HAYABUSA_AMICA for the Itokawa simulation

  # -aperture 0 ignored 1 0 \
  # ignored

  # -pixel_angle 0 unit \
  # ingnored

  # -distortion 0 0.0 0.0 0.0 0.0 0.0 \
  # ignored

  # tangential 0 0.0 0.0 0.0 0.0 \
  # ignored

  # scattering 0 ignored 0.0 0.0 0.0 0 \
  # ignored

  # -psf_gauss 0 0 0.0 0 0.0 0 0.0 \
  # ignored

  # -exposure 0 0.000015 300 \
  # Temperature 300K, exposure 15us as per DVS paper

  # -readout_basic 0 0 1.0 0.0 1.0 1e6 55 24966 \
  # # bloom - off
  # # qe = 1 (this is definitely wrong)
  # # bias = 0 (no extra electron besides dark current)
  # # gain = 1 (to get back electron count)
  # # well = 1e6 (big so it doesn't saturate)
  # # rms = 55e (taken from DAVIS APS datasheet)
  # # dc = 24966 [e/s] (electron per s converted from Idark from DVS paper)

  # -smear 0 none down 1.0 0.0 0.0 0.0 \
  # ignored

  # -photon_noise 0 1 \
  # Add photon shot noise

  # -dark_current 0 constant 0 1 \
  # # We know the dark current, so we leave it as constant

  # -detector_radiation 0 0 \
  # ignored



  -use_camera_model \
  \
  -detector 0 1 128 128 0 \
  -noinverse_square_law \
  -irradiance 1.0 1.0 1.0 \
  -aperture 0 ignored 1 0 \
  -pixel_angle 0 unit \
  -distortion 0 0.0 0.0 0.0 0.0 0.0 \
  -tangential 0 0.0 0.0 0.0 0.0 \
  -scattering 0 ignored 0.0 0.0 0.0 0 \
  -psf_gauss 0 0 0.0 0 0.0 0 0.0 \
  -exposure 0 0.15 300 \
  -readout_basic 0 0 0.02 0.0 1.0 1e12 55 24966 \
  -smear 0 none down 1.0 0.0 0.0 0.0 \
  -photon_noise 0 1 \
  -dark_current 0 none 0 1 \
  -detector_radiation 0 0 \
  \
  -flight test_traj.fli \
  -movie \
  -image_format float \
  -savefmt frames/test2/test2_%01d \
