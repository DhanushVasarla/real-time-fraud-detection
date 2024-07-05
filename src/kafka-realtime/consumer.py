from confluent_kafka import Consumer
import socket
import time
import pickle
import json
import os
import sys
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CustomTransformers import FeatureAdder

# sys.path.append(os.path.abspath('..','train.py'))
pipeline_path =  os.path.join('..','models','new_pipeline.pkl')

with open(pipeline_path, 'rb') as f:
    loaded_pipeline = pickle.load(f)

cols = ["distance_from_home","distance_from_last_transaction","ratio_to_median_purchase_price","repeat_retailer","used_chip","used_pin_number","online_order"]

def make_inference_single_record(values):
    record_dict = {column:value for column,value in zip(cols,values)}
    single_record = pd.DataFrame(record_dict,index=[0])
    pred = loaded_pipeline.predict(single_record)
    return list(pred)[0]

def oauth_cb(oauth_config):
    auth_token, expiry_ms = MSKAuthTokenProvider.generate_auth_token("ap-south-1")
    return auth_token, expiry_ms/1000

c = Consumer({
    
    'bootstrap.servers': "b-2.kafka.16gabb.c3.kafka.ap-south-1.amazonaws.com:9098",
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
    msg = c.poll(timeout=1.0)

    if msg is None:
        continue
    if msg.error():
        print("Consumer error: {}".format(msg.error()))
        continue

    message_str = msg.value().decode('utf-8')
    message_list = json.loads(message_str)
    print(message_list)
    prediction = make_inference_single_record(message_list)
    print(prediction)

    if prediction == 0:
        print("Not Fraud")
    else:
        print("Detected Fraud!")
    # print(f"Received: {message_dict}")
    # print(type(message_dict))


c.close()