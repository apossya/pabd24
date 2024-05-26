"""Upload selected files to S3 storage"""
import argparse

from dotenv import dotenv_values
import boto3

BUCKET_NAME = 'pabd24'
YOUR_ID = '13'
CSV_PATH = ['C:/Users/toni2/PycharmProjects/pabd2024_Posenitskiy/pabd24/data/raw/1_2024-05-26_18-22.csv',
            'C:/Users/toni2/PycharmProjects/pabd2024_Posenitskiy/pabd24/data/raw/2_2024-05-26_18-23.csv',
            'C:/Users/toni2/PycharmProjects/pabd2024_Posenitskiy/pabd24/data/raw/3_2024-05-26_18-23.csv']

config = dotenv_values(".env")
client = boto3.client(
    's3',
    endpoint_url='https://storage.yandexcloud.net',
    aws_access_key_id=config['KEY'],
    aws_secret_access_key=config['SECRET']
)


def main(args):
    for csv_path in args.input:
        remote_name = f'{YOUR_ID}/' + csv_path.replace('\\', '/')
        client.download_file(BUCKET_NAME, remote_name, csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+',
                        help='Remote data files to download from S3 storage',
                        default=CSV_PATH)
    args = parser.parse_args()
    main(args)