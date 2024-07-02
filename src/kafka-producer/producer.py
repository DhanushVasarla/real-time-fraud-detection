from confluent_kafka import Producer
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
import socket
import json
import time

class MSKTokenProvider:
    def __init__(self, region):
        self.region = region

    def token(self):
        token, _ = MSKAuthTokenProvider.generate_auth_token(self.region)
        return token

tp = MSKTokenProvider(region='ap-south-1')


def oauth_callback():
    return tp.token()

conf = {
    'bootstrap.servers': 'b-1.mk0kafka.bfqse5.c3.kafka.ap-south-1.amazonaws.com:9098',
    'security.protocol': 'SASL_SSL',
    'sasl.mechanism': 'OAUTHBEARER',
    'sasl.oauthbearer.method': 'default',
    'sasl.oauthbearer.config': 'scope=myscope',  
    'client.id': socket.gethostname(),
}

producer = Producer(**conf)

topic = "kafka-topic-1"
ctr = 1

def delivery_report(err, msg):
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

try:
    while True:
        inp = str(ctr)
        message = json.dumps({"message": inp}).encode('utf-8')
        producer.produce(topic, key=str(ctr), value=message, on_delivery=delivery_report)
        producer.poll(0)
        print("Produced message:", inp)
        ctr += 1
        time.sleep(1)  # Sleep for a second before producing the next message
except KeyboardInterrupt:
    print("Producer stopped.")
except Exception as e:
    print(f"Failed to send message: {e}")
finally:
    producer.flush()  # Ensure all messages are delivered