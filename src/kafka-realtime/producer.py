from kafka import KafkaProducer
from kafka.errors import KafkaError
import socket
import time
import random
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider

class MSKTokenProvider():
    def token(self):
        token, _ = MSKAuthTokenProvider.generate_auth_token('ap-south-1')
        return token

tp = MSKTokenProvider()

producer = KafkaProducer(
    bootstrap_servers='b-1.kafka.xy5s1w.c3.kafka.ap-south-1.amazonaws.com:9098',
    security_protocol='SASL_SSL',
    sasl_mechanism='OAUTHBEARER',
    sasl_oauth_token_provider=tp,
    client_id=socket.gethostname(),
)

topic = "kafka-topic-2"

def generate_random_message():
    return [
        random.uniform(0, 1000),  
        random.uniform(0, 1000),
        random.uniform(0, 1000),
        random.uniform(0, 1),    
        random.uniform(0, 1),
        random.choice([0, 1]),   
        random.choice([0, 1]),
        random.choice([0, 1])
    ]


while True:
    try:
        msg = generate_random_message()
        inp=str(msg)
        producer.send(topic, inp.encode())
        producer.flush()
        print(f"Produced: {inp}")
        time.sleep(0)
    except Exception:
        print("Failed to send message:", e)

producer.close()