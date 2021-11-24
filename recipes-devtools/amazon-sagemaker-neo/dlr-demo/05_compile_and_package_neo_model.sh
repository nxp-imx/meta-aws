if [ ! ${AWS_REGION} ]; then
  echo "Please set AWS_REGION..."
  exit 1
fi
if [ ! ${ROLE_ARN} ]; then
  echo "Please set ROLE_ARN..."
  exit 1
fi
if [ ! ${MODULE_URL} ]; then
  echo "Please set MODULE_URL..."
  exit 1
fi

COMPONENTS_BUCKET_NAME=${PROJECT_NAME}-greengrass-components
COMPILATION_JOB_NAME=$(uuidgen)-mobilenet-v2

#Upload model to components bucket
wget ${MODULE_URL} -O mobilenet_v2_1.0_224_quanta.tgz
aws s3 cp mobilenet_v2_1.0_224_quanta.tgz s3://${COMPONENTS_BUCKET_NAME}/models/uncompiled/mobilenet_v2_1.0_224_quanta.tgz
echo "Model has been uploaded to s3"

#Create neo job to compile the model
aws sagemaker create-compilation-job --compilation-job-name ${COMPILATION_JOB_NAME} --role-arn ${ROLE_ARN} \
  --input-config "{\"S3Uri\": \"s3:\/\/${COMPONENTS_BUCKET_NAME}\/models\/uncompiled\/mobilenet_v2_1.0_224_quanta.tgz\",\"DataInputConfig\": \"{\\\"input\\\":[1,224,224,3]}\",\"Framework\": \"TFLITE\"}" \
  --output-config "{\"S3OutputLocation\":\"s3:\/\/${COMPONENTS_BUCKET_NAME}\/models\/compiled\/\",\"TargetDevice\":\"imx8mplus\"}" \
  --stopping-condition "{\"MaxWaitTimeInSeconds\":60,\"MaxRuntimeInSeconds\":900}" --region $AWS_REGION

echo "Neo compilation job created, the job will take 5 minuts."
while [ "${CJOB_STATUS}" != "COMPLETED" ]
do
  echo "Waiting compilation job to complete"
  CJOB_STATUS=$(aws sagemaker describe-compilation-job --compilation-job-name ${COMPILATION_JOB_NAME} | jq -r .CompilationJobStatus)
  sleep 30
done


PACKAGING_JOB_NAME=$(uuidgen)-mobilenet-v2
MODEL_COMPONENT_NAME=${PROJECT_NAME}_mobilenetv2_224_quant_model
MODEL_NAME=mobilenetv2-224-10-quant

#Create model packageing job
aws sagemaker create-edge-packaging-job --edge-packaging-job-name ${PACKAGING_JOB_NAME} \
  --compilation-job-name ${COMPILATION_JOB_NAME} --model-name ${MODEL_NAME} --model-version 1.0 \
  --role-arn ${ROLE_ARN} --output-config "{\"S3OutputLocation\":\"s3:\/\/${COMPONENTS_BUCKET_NAME}\/models\/packaged\/\",\"PresetDeploymentType\":\"GreengrassV2Component\",\"PresetDeploymentConfig\":\"{\\\"ComponentName\\\":\\\"${MODEL_COMPONENT_NAME}\\\",\\\"ComponentDescription\\\":\\\"${PROJECT_NAME} mobilenetv2_224_quant_model\\\",\\\"ComponentVersion\\\":\\\"1.0.0\\\"}\"}" \
  --region $AWS_REGION

echo "Neo packaging job created, the job will take 2 minuts."
while [ "${PJOB_STATUS}" != "COMPLETED" ]
do
  echo "Waiting packaging job to complete"
  PJOB_STATUS=$(aws sagemaker describe-edge-packaging-job --edge-packaging-job-name ${PACKAGING_JOB_NAME} \
    | jq -r .EdgePackagingJobStatus )
  sleep 30
done

MODEL_COMPONENT_ARN=$(aws sagemaker describe-edge-packaging-job \
  --edge-packaging-job-name ${PACKAGING_JOB_NAME} | jq -r .PresetDeploymentOutput.Artifact)
