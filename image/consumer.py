#!/Users/navanath.navaskar/rabbitmqenv/bin/python3
# -*- coding=utf-8  
# pip install -U cos-python-sdk-v5
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import os
import pika
import threading
from run import task_app
import argparse
import functools
import json

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--config_path', default="config/configs_32/*.json", type=str, help='config file path')
parser.add_argument('--base_dataset_dir', default="/home/mengyang/dataset/images/", type=str, help='base dataset dir')
parser.add_argument('--bath_model_dir', default="/home/mengyang/dataset/images_model/", type=str, help='bath model dir')
parser.add_argument('--output_dir', default="/home/mengyang/dataset/images_td/", type=str, help='output dir')
parser.add_argument('--queue', default="configs_32", type=str, help='queue name')
args = parser.parse_args()


# 连接cos
secret_id = '*'     
secret_key = '*'   
config = CosConfig(Region='ap-beijing', SecretId=secret_id, SecretKey=secret_key, Token=None, Scheme='https')
client = CosS3Client(config)

def cos_upload_file(path_file,dirs="test"):
    file_name = os.path.basename(path_file)
    client.upload_file(Bucket='task-1254003045',LocalFilePath=path_file,Key=os.path.join(dirs,file_name),PartSize=1,MAXThread=10,EnableMD5=False)


# def callback(ch, method, properties, body):
#     try:
#         # 保存body.decode到本地文件temp.json
#         data = json.loads(body.decode())
#         task_app(data, args.base_dataset_dir, args.bath_model_dir, args.output_dir)
#         print(f"success processing")
#         # 消息成功处理后，确认消息已被消费
#         ch.basic_ack(delivery_tag=method.delivery_tag)
        
#     except Exception as e:
#         # 处理异常，可以将消息重新放回队列或者记录日志
#         print(f"Error processing message: {str(e)}")
#         ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    

def ack_message(channel, delivery_tag):
    print(f'ack_message thread id: {threading.get_ident()}')
    if channel.is_open:
        channel.basic_ack(delivery_tag)
    else:
        # Channel is already closed, so we can't ACK this message;
        # log and/or do something that makes sense for your app in this case.
        pass


def do_work(channel, delivery_tag, body):
    print(f'do_work thread id: {threading.get_ident()}')
    data = json.loads(body.decode())
    task_app(data, args.base_dataset_dir, args.bath_model_dir, args.output_dir)
    cb = functools.partial(ack_message, channel, delivery_tag)
    channel.connection.add_callback_threadsafe(cb)


def on_message(channel, method_frame, header_frame, body):
    print(f'on_message thread id: {threading.get_ident()}')
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(channel, delivery_tag, body))
    t.start()



connection = pika.BlockingConnection(pika.ConnectionParameters('39.99.241.32', 5672, '/', pika.PlainCredentials('lmy', '***'), heartbeat=0))
channel = connection.channel()
channel.queue_declare(queue=args.queue, durable=True)
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=args.queue, on_message_callback=on_message)
channel.start_consuming()