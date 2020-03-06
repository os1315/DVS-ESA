#!/bin/sh

# Variables from Python
NEW_FILE=$1
TARG_DIR=$2

# Bright params
EXP1=$3
QE1=$4
BIAS1=$5
GAIN1=$6
RMS1=$7
DC1=$8

# Dim params
EXP2=$9
QE2=${10}
BIAS2=${11}
GAIN2=${12}
RMS2=${13}
DC2=${14}

echo $TARG_DIR
cd $TARG_DIR


NEW_PATH="frames/${NEW_FILE}"

if [ -d $NEW_PATH ]
then
  echo "A folder for $NEW_FILE exists"
  echo "I am set to exit to prevent overwrite, change singleNB.sh in order to override that."
  echo " "
  exit
else
  echo "Creating new directory for $NEW_FILE"
  mkdir $NEW_PATH
  cp test_traj.fli $NEW_PATH
  NEW_PATH="frames/${NEW_FILE}/raw_bright"
  NEW_CMD="frames/${NEW_FILE}/raw_bright/${NEW_FILE}_%03d"
  mkdir $NEW_PATH
fi

# Basic view script for Itokawa model using standard radiance camera.
../../bin/viewer \
	-noini \
	-ini ../pangu.ini \
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
  -detector 0 1 128 128 0 \
  -irradiance 10000000.0 10000000.0 10000000.0 \
  -inverse_square_law \
  -aperture 0 ignored 1 0 \
  -pixel_angle 0 unit \
  -distortion 0 0.0 0.0 0.0 0.0 0.0 \
  -tangential 0 0.0 0.0 0.0 0.0 \
  -scattering 0 ignored 0.0 0.0 0.0 0 \
  -psf_gauss 0 0 0.0 0 0.0 0 0.0 \
  -exposure 0 $EXP1 300 \
  -readout_basic 0 0 $QE1 $BIAS1 $GAIN1 1e12 $RMS1 $DC1 \
  -smear 0 none down 1.0 0.0 0.0 0.0 \
  -photon_noise 0 0 \
  -dark_current 0 none 0 0 \
  -detector_radiation 0 0 \
  \
  -flight test_traj.fli \
  -movie \
  -image_format float \
  -savefmt "frames/${NEW_FILE}/raw_bright/${NEW_FILE}_%03d" \
  -quit \
  model.pan \
  \

NEW_PATH="frames/${NEW_FILE}/raw_dim"
mkdir $NEW_PATH

# Basic view script for Itokawa model using standard radiance camera.
../../bin/viewer \
	-noini \
	-ini ../pangu.ini \
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
  -detector 0 1 128 128 0 \
  -irradiance 10000000.0 10000000.0 10000000.0 \
  -inverse_square_law \
  -aperture 0 ignored 1 0 \
  -pixel_angle 0 unit \
  -distortion 0 0.0 0.0 0.0 0.0 0.0 \
  -tangential 0 0.0 0.0 0.0 0.0 \
  -scattering 0 ignored 0.0 0.0 0.0 0 \
  -psf_gauss 0 0 0.0 0 0.0 0 0.0 \
  -exposure 0 $EXP2 300 \
  -readout_basic 0 0 $QE2 $BIAS2 $GAIN2 1e12 $RMS2 $DC2 \
  -smear 0 none down 1.0 0.0 0.0 0.0 \
  -photon_noise 0 0 \
  -dark_current 0 constant 0 0 \
  -detector_radiation 0 0 \
  \
  -flight test_traj.fli \
  -movie \
  -image_format float \
  -savefmt "frames/${NEW_FILE}/raw_dim/${NEW_FILE}_%03d" \
  -quit \
  model.pan \
  \
