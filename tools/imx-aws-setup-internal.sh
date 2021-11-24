#!/bin/sh
#
# Copyright 2021 NXP

. fsl-setup-internal-build.sh $@

BUILD_DIR=.

if [ `grep -c "meta-aws" $BUILD_DIR/conf/bblayers.conf` -ne '0' ];then
  echo "Skip adding AWS layer"
else
  echo "Adding AWS layer"
  echo "#For AWS" >> $BUILD_DIR/conf/bblayers.conf
  echo "BBLAYERS += \"\${BSPDIR}/sources/meta-aws\"" >> $BUILD_DIR/conf/bblayers.conf

  echo "#For AWS" >> $BUILD_DIR/conf/local.conf
  echo "IMAGE_INSTALL_append = \" greengrass-bin\"" >> $BUILD_DIR/conf/local.conf
  echo "IMAGE_INSTALL_append = \" aws-iot-device-sdk-python-v2\"" >> $BUILD_DIR/conf/local.conf
  echo "IMAGE_INSTALL_append = \" neo-ai-dlr\"" >> $BUILD_DIR/conf/local.conf
  echo "IMAGE_INSTALL_append = \" dlr-demo\"" >> $BUILD_DIR/conf/local.conf
  echo "PREFERRED_VERSION_tim-vx = \"1.1.30\"" >> $BUILD_DIR/conf/local.conf
  echo "PREFERRED_VERSION_python3-botocore = \"1.22.12\"" >> $BUILD_DIR/conf/local.conf
  echo "PREFERRED_VERSION_python3-boto3 = \"1.19.11\"" >> $BUILD_DIR/conf/local.conf
fi
