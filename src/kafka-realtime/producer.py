from kafka import KafkaProducer
from kafka.errors import KafkaError
import socket
import time
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider

class MSKTokenProvider():
    def token(self):
        token, _ = MSKAuthTokenProvider.generate_auth_token('ap-south-1')
        return token

tp = MSKTokenProvider()

producer = KafkaProducer(
    bootstrap_servers='b-1.kafka.1es57u.c3.kafka.ap-south-1.amazonaws.com:9098,b-2.kafka.1es57u.c3.kafka.ap-south-1.amazonaws.com:9098',
    security_protocol='SASL_SSL',
    sasl_mechanism='OAUTHBEARER',
    sasl_oauth_token_provider=tp,
    client_id=socket.gethostname(),
)

topic = "kafka-topic"
msg = "[57.87785658389723,0.3111400080477545,1.9459399775518593,1.0,1.0,0.0,0.0,0.0]"
while True:
    try:
        inp=str(msg)
        producer.send(topic, inp.encode())
        producer.flush()
        print("Produced!")
        time.sleep(2)
        # ctr+=1
    except Exception:
        print("Failed to send message:", e)

producer.close()