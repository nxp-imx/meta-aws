SUMMARY = "Scripts to setup sagemaker and greengrass"
DESCRIPTION = "Scripts to setup sagemaker and greengrass"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=d8e55e38068e35fc7fb72940c3c077c0"


SRC_URI = "file://LICENSE \
           file://00_setup_cloud_service.sh \
           file://01_create_greengrass_core.sh \
           file://02_create_greengrass_role.sh \
           file://03_upload_component_version.sh \
           file://04_create_device_fleet_register_device.sh \
           file://05_compile_and_package_neo_model.sh \
           file://06_create_greengrass_deployment.sh \
           file://07_setup_device_greengrass.sh \
           file://10_clean_up.sh \
           file://setup_cloud_service_and_device.sh \
           file://device-role-access-policy.json \
           file://device-role-trust-policy.json \
           file://greengrass-v2-iot-policy.json \
           file://greengrass_config.yaml \
           file://check_result.py \
           file://agent_pb2_grpc.py \
           file://agent_pb2.py \
           file://README \
           file://aws.sagemaker.edgeManagerClientCameraIntegration-0.1.0.yaml \
           file://camera_integration_edgemanger_client.py \
           "

S = "${WORKDIR}"

do_install() {
    install -d ${D}${bindir}/${PN}-scripts
    install -d ${D}${bindir}/${PN}-scripts/components

    # Now install python test scripts
    install -m 0755 ${S}/*.sh ${D}${bindir}/${PN}-scripts
    install -m 0644 ${S}/check_result.py ${D}${bindir}/${PN}-scripts
    install -m 0644 ${S}/*.json ${D}${bindir}/${PN}-scripts
    install -m 0644 ${S}/README ${D}${bindir}/${PN}-scripts
    install -m 0644 ${S}/greengrass_config.yaml ${D}${bindir}/${PN}-scripts

    install -m 0644 ${S}/agent_pb2_grpc.py ${D}${bindir}/${PN}-scripts/components
    install -m 0644 ${S}/agent_pb2.py ${D}${bindir}/${PN}-scripts/components
    install -m 0644 ${S}/camera_integration_edgemanger_client.py ${D}${bindir}/${PN}-scripts/components
    install -m 0644 ${S}/aws.sagemaker.edgeManagerClientCameraIntegration-0.1.0.yaml ${D}${bindir}/${PN}-scripts/components
}

RDEPENDS_${PN} += "awscli (>=1.21.12)\
                   neo-ai-dlr \
                   curl \
                   jq \
                   python3-pexpect \
                   greengrass-bin (>=2.4.0) \
                   "

# Output library is unversioned
FILES_SOLIBSDEV = ""

COMPATIBLE_MACHINE          = "(^$)"
COMPATIBLE_MACHINE_imxgpu3d = "(mx8)"
COMPATIBLE_MACHINE_mx8mm    = "(^$)"

