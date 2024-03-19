#!/Users/navanath.navaskar/rabbitmqenv/bin/python3

import pika
import os
import json
import glob



def send_task(queue, task_dir):
    connection = pika.BlockingConnection(pika.ConnectionParameters('39.99.241.32', 5672, '/', pika.PlainCredentials('lmy', '***'), heartbeat=0))
    channel = connection.channel()
    channel.queue_declare(queue=queue, durable=True)
    # get message
    for file in glob.glob(f"{task_dir}/*.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            channel.basic_publish(exchange='',routing_key=queue,body=json.dumps(data),properties=pika.BasicProperties(delivery_mode=2))
            print(" [x] Sent %r" % data)
    connection.close()


if __name__ == "__main__":
    # send_task("configs_32", "config/configs_32")
    send_task("configs_224_1", "config/configs_224_1")
    send_task("configs_224_2", "config/configs_224_2")
    send_task("configs_224_3", "config/configs_224_3")
    send_task("configs_224_4", "config/configs_224_4")