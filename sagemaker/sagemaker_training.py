"""
Setup and trigger SageMaker training + batch prediction jobs (NEW API)
"""

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

REGION = "ap-southeast-2"
ROLE_ARN = "arn:aws:iam::248896561752:role/service-role/AmazonSageMakerAdminIAMExecutionRole"

PYTORCH_IMAGE = (
    f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.0.0-cpu-py310"
)


def upload_data_to_s3(local_csv_path, s3_prefix='datasets'):
    s3 = boto3.client('s3')
    bucket_name = "amazon-sagemaker-248896561752-ap-southeast-2-c3asvvi6hbt3qa"
    s3_key = f'{s3_prefix}/items.csv'

    print(f'Uploading {local_csv_path} to s3://{bucket_name}/{s3_key}')
    s3.upload_file(local_csv_path, bucket_name, s3_key)

    return f's3://{bucket_name}/{s3_prefix}'


def train_shop_model(
    shop_name,
    bucket_name,
    training_script='sagemaker/training_script.py',
    instance_type='ml.m5.large'
):
    sagemaker_session = sagemaker.Session()
    data_s3_path = f's3://{bucket_name}/datasets'

    estimator = Estimator(
        image_uri=PYTORCH_IMAGE,
        entry_point=training_script,
        role=ROLE_ARN,
        instance_count=1,
        instance_type=instance_type,
        hyperparameters={
            'epochs': '100',
            'batch-size': '32',
            'learning-rate': '0.001',
            'hidden-size': '64',
            'num-layers': '2',
            'seq-length': '5',
            'shop-name': shop_name
        },
        output_path=f's3://{bucket_name}/models/{shop_name}/',
        sagemaker_session=sagemaker_session,
        tags=[
            {'Key': 'Shop', 'Value': shop_name},
            {'Key': 'ModelType', 'Value': 'ItemPrediction'}
        ]
    )

    print(f'Starting training job for {shop_name} shop...')
    estimator.fit({'train': data_s3_path}, wait=True)

    print(f'✓ Training complete for {shop_name}!')
    print(f'✓ Model artifacts: {estimator.model_data}')

    return estimator


def generate_shop_predictions(
    shop_name,
    model_s3_path,
    bucket_name,
    cycle_minutes,
    inference_script='sagemaker/inference_script.py',
    batch_script='sagemaker/batch_predictions.py',
    instance_type='ml.m5.large'
):

    sagemaker_session = sagemaker.Session()

    processor = ScriptProcessor(
        role=ROLE_ARN,
        image_uri=PYTORCH_IMAGE,
        instance_count=1,
        instance_type=instance_type,
        command=['python3'],
        sagemaker_session=sagemaker_session
    )

    print(f'Starting prediction generation for {shop_name}...')

    processor.run(
        code=batch_script,
        inputs=[
            ProcessingInput(
                source=model_s3_path,
                destination='/opt/ml/processing/model',
                input_name='model'
            ),
            ProcessingInput(
                source=f's3://{bucket_name}/datasets/{shop_name}_item_deltas.csv',
                destination='/opt/ml/processing/input',
                input_name='data'
            )
        ],
        outputs=[
            ProcessingOutput(
                source='/opt/ml/processing/output',
                destination=f's3://{bucket_name}/predictions',
                output_name='predictions'
            )
        ],
        arguments=[
            '--shop-name', shop_name,
            '--cycle-minutes', str(cycle_minutes),
            '--model-path', '/opt/ml/processing/model',
            '--data-path', '/opt/ml/processing/input',
            '--output-path', '/opt/ml/processing/output'
        ],
        wait=True
    )

    predictions_s3_path = f's3://{bucket_name}/predictions/{shop_name}_predictions.csv'
    print(f'✓ Predictions generated: {predictions_s3_path}')

    return predictions_s3_path


def train_weather_model(
    bucket_name,
    training_script='sagemaker/weather_training_script.py',
    instance_type='ml.m5.large'
):
    sagemaker_session = sagemaker.Session()
    data_s3_path = f's3://{bucket_name}/datasets'

    estimator = Estimator(
        image_uri=PYTORCH_IMAGE,
        entry_point=training_script,
        role=ROLE_ARN,
        instance_count=1,
        instance_type=instance_type,
        hyperparameters={
            'epochs': '100',
            'batch-size': '32',
            'learning-rate': '0.001',
            'hidden-size': '64',
            'num-layers': '2',
            'seq-length': '5'
        },
        output_path=f's3://{bucket_name}/models/weather/',
        sagemaker_session=sagemaker_session
    )

    print('Starting weather model training...')
    estimator.fit({'train': data_s3_path}, wait=True)

    print('✓ Weather model training complete!')
    print(f'✓ Model artifacts: {estimator.model_data}')

    return estimator


def generate_weather_predictions(
    model_s3_path,
    bucket_name,
    batch_script='sagemaker/batch_predictions.py',
    instance_type='ml.m5.large'
):
    from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

    sagemaker_session = sagemaker.Session()

    processor = ScriptProcessor(
        role=ROLE_ARN,
        image_uri=PYTORCH_IMAGE,
        instance_count=1,
        instance_type=instance_type,
        command=['python3'],
        sagemaker_session=sagemaker_session
    )

    print('Starting weather prediction generation...')

    processor.run(
        code=batch_script,
        inputs=[
            ProcessingInput(
                source=model_s3_path,
                destination='/opt/ml/processing/model',
                input_name='model'
            ),
            ProcessingInput(
                source=f's3://{bucket_name}/datasets/weather_deltas.csv',
                destination='/opt/ml/processing/input',
                input_name='data'
            )
        ],
        outputs=[
            ProcessingOutput(
                source='/opt/ml/processing/output',
                destination=f's3://{bucket_name}/predictions',
                output_name='predictions'
            )
        ],
        arguments=[
            '--weather',
            '--model-path', '/opt/ml/processing/model',
            '--data-path', '/opt/ml/processing/input',
            '--output-path', '/opt/ml/processing/output'
        ],
        wait=True
    )

    predictions_s3_path = f's3://{bucket_name}/predictions/weather_predictions.csv'
    print(f'✓ Weather predictions generated: {predictions_s3_path}')

    return predictions_s3_path
