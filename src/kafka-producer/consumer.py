from confluent_kafka import Consumer
import socket
import time
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider

def oauth_cb(oauth_config):
    auth_token, expiry_ms = MSKAuthTokenProvider.generate_auth_token("ap-south-1")
    # Note that this library expects oauth_cb to return expiry time in seconds since epoch, while the token generator returns expiry in ms
    return auth_token, expiry_ms / 1000

conf = {
    'bootstrap.servers': "b-1.mk0kafka.bfqse5.c3.kafka.ap-south-1.amazonaws.com:9098",
    'client.id': socket.gethostname(),
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'OAUTHBEARER',
    'sasl.oauthbearer.method': 'default',
    'sasl.oauthbearer.config': 'scope=myscope',
    'group.id': 'mygroup',
    'auto.offset.reset': 'earliest',
    'socket.timeout.ms': 60000,
    'oauth_cb': oauth_cb
}

consumer = Consumer(**conf)
consumer.subscribe(['kafka-topic-1'])

print("Starting consumer!")

try:
    while True:
        msg = consumer.poll(5)

        if msg is None:
            continue
        if msg.error():
            print("Consumer error: {}".format(msg.error()))
            continue
        print('Received message: {}'.format(msg.value().decode('utf-8')))
except KeyboardInterrupt:
    print("Consumer stopped.")
finally:
    consumer.close()