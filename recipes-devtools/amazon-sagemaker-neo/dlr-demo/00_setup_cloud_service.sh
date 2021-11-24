MODULE_URL=https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
TING_CAMERA_DEVICE=3

aws configure set region ${AWS_REGION}
if [ ! ${PROJECT_NAME} ]; then
  echo "Please set PROJECT_NAME(one uniqe string with lowercase letters only)..."
  exit 1
fi

if [ ${CAMERA_DEVICE} ]; then
  TING_CAMERA_DEVICE=${CAMERA_DEVICE}
fi
echo "The camera device is ${TING_CAMERA_DEVICE}"

echo "PROJECT_NAME=${PROJECT_NAME}" > project_config.sh
echo "AWS_REGION=${AWS_REGION}" >> project_config.sh

source ./01_create_greengrass_core.sh
echo "THING_NAME=${THING_NAME}" >> project_config.sh
echo "CERT_ARN=${CERT_ARN}" >> project_config.sh
echo "THING_GROUP=${THING_GROUP}" >> project_config.sh
echo "THING_POLICY=${THING_POLICY}" >> project_config.sh
echo "IOT_DATA_ENDPOINT=${IOT_DATA_ENDPOINT}" >> project_config.sh
echo "IOT_CRED_ENDPOINT=${IOT_CRED_ENDPOINT}" >> project_config.sh

source ./02_create_greengrass_role.sh
echo "ROLE_ALIAS_NAME=${ROLE_ALIAS_NAME}" >> project_config.sh
echo "ROLE_ALIAS_POLICY_NAME=${ROLE_ALIAS_POLICY_NAME}" >> project_config.sh
echo "ROLE_NAME=${ROLE_NAME}" >> project_config.sh
echo "ROLE_POLICY_ARN=${ROLE_POLICY_ARN}" >> project_config.sh

source ./03_upload_component_version.sh
echo "CAMERA_COMPONENT_ARN=${CAMERA_COMPONENT_ARN}" >> project_config.sh
echo "COMPONENTS_BUCKET_NAME=${COMPONENTS_BUCKET_NAME}" >> project_config.sh

source ./04_create_device_fleet_register_device.sh
echo "DEVICE_FLEET_NAME=${DEVICE_FLEET_NAME}" >> project_config.sh
echo "DEVICE_NAME=${DEVICE_NAME}" >> project_config.sh
echo "RESULTS_BUCKET_NAME=${RESULTS_BUCKET_NAME}" >> project_config.sh

source ./05_compile_and_package_neo_model.sh
echo "MODEL_COMPONENT_ARN=${MODEL_COMPONENT_ARN}" >> project_config.sh

source ./06_create_greengrass_deployment.sh
echo "DEPLOYMENT_ID=${DEPLOYMENT_ID}" >> project_config.sh
