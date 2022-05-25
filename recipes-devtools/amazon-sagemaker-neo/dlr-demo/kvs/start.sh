
#!/bin/sh

export AWS_IOT_CORE_THING_NAME=bj03-gg-webrtc_GreengrassCore_0a4864c8
#export AWS_IOT_CORE_CREDENTIAL_ENDPOINT=$(aws iot describe-endpoint --endpoint-type iot:CredentialProvider --output text)
export AWS_IOT_CORE_CREDENTIAL_ENDPOINT=c30j1jnpckgt7a.credentials.iot.us-east-2.amazonaws.com
export AWS_IOT_CORE_ROLE_ALIAS=bj03-gg-webrtc_GreengrassCoreRoleAlias

export AWS_DEFAULT_REGION=us-east-2
export AWS_KVS_CACERT_PATH=certs/cert.pem
export AWS_IOT_CORE_CERT=/greengrass/v2/device.pem.crt
export AWS_IOT_CORE_PRIVATE_KEY=/greengrass/v2/private.pem.key

/kvs/kvsWebrtcClientMasterGstSample $AWS_IOT_CORE_THING_NAME video-only testsrc
