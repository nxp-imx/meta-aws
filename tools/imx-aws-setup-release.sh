#!/bin/sh
#
# Copyright 2021 NXP

. imx-setup-release.sh $@

BUILD_DIR=.
echo "Adding Aws layer"
echo "" >> $BUILD_DIR/conf/bblayers.conf
echo "# For Aws" >> $BUILD_DIR/conf/bblayers.conf
echo "BBLAYERS += \"\${BSPDIR}/sources/meta-aws\"" >> $BUILD_DIR/conf/bblayers.conf
echo "IMAGE_INSTALL_append = \" greengrass-bin\"" >> $BUILD_DIR/conf/local.conf
echo "IMAGE_INSTALL_append = \" aws-iot-device-sdk-python-v2\"" >> $BUILD_DIR/conf/local.conf
echo "IMAGE_INSTALL_append = \" neo-ai-dlr\"" >> $BUILD_DIR/conf/local.conf
echo "IMAGE_INSTALL_append = \" awscli\"" >> $BUILD_DIR/conf/local.conf
echo "PREFERRED_VERSION_tim-vx = \"1.1.30\"" >> $BUILD_DIR/conf/local.conf
