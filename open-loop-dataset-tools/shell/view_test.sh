#!/bin/sh

# Basic view script for Itokawa model using standard radiance camera.
../../bin/viewer \
	-noini \
	-ini ../pangu.ini \
	-ini itokawa.ini \
	-err - \
	\
	-colour 1 1 1 \
	-sky black \
	-dynamic_shadows none \
	-nocull_face \
	\
	-reflect hapke \
	-hapke_w     0.33 \
	-hapke_B0    0.95 \
	-hapke_h     0.05 \
	-hapke_L     0.000 \
	-hapke_scale 4.667314 \
  \
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
  itokawa_q512.pan \
  \
  $*
