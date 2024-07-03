from confluent_kafka import Consumer
import socket
import time
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider

def oauth_cb(oauth_config):
    auth_token, expiry_ms = MSKAuthTokenProvider.generate_auth_token("ap-south-1")
    # Note that this library expects oauth_cb to return expiry time in seconds since epoch, while the token generator returns expiry in ms
    return auth_token, expiry_ms/1000

c = Consumer({
    
    'bootstrap.servers': "b-1.kafka.1es57u.c3.kafka.ap-south-1.amazonaws.com:9098,b-2.kafka.1es57u.c3.kafka.ap-south-1.amazonaws.com:9098",
    'client.id': socket.gethostname(),
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'OAUTHBEARER',
    'oauth_cb': oauth_cb,
    'group.id': 'mygroup',
    'auto.offset.reset': 'earliest'
})

c.subscribe(['kafka-topic'])

print("Starting consumer!")

while True:
    msg = c.poll(5)

    if msg is None:
        continue
    if msg.error():
        print("Consumer error: {}".format(msg.error()))
        continue
    print('Received message: {}'.format(msg.value().decode('utf-8')))

c.close()