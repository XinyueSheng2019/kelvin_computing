import json
from lasair import lasair_consumer

kafka_server = 'kafka.lsst.ac.uk:9092'
group_id     = 'xys_01'
my_topic     = 'lasair_2SN-likecandidates'
consumer = lasair_consumer(kafka_server, group_id, my_topic)
import json
n = 0
while n < 10:
    msg = consumer.poll(timeout=20)
    if msg is None:
        break
    if msg.error():
        print(str(msg.error()))
        break
    jmsg = json.loads(msg.value())
    print(json.dumps(jmsg, indent=2))
    n += 1
print('No more messages available')