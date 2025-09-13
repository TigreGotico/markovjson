from markovjson.topic_modelling import MarkovTopic
from os.path import join, dirname
from os import listdir
from pprint import pprint

folder = join(dirname(__file__), "intents")

container = MarkovTopic(order=4)

for f in listdir(folder):
    if f.endswith(".txt"):
        container.register_topic_from_file(join(folder, f))

utt = "turn off"

a = container.score_tokens(utt)
pprint(a)
# {'[/LABEL=hello.txt]': {'off': 0.0, 'turn': 0.0},
#  '[/LABEL=joke.txt]': {'off': 0.0, 'turn': 0.0},
#  '[/LABEL=lights_off.txt]': {'off': 1.0, 'turn': 0.6666666666666666},
#  '[/LABEL=lights_on.txt]': {'off': 0.0, 'turn': 0.6666666666666666},
#  '[/LABEL=thank.txt]': {'off': 0.0, 'turn': 0.0}}

pprint(container.predict_topic(utt))
# {'[/LABEL=lights_off.txt]': 0.8333333333333333,
#  '[/LABEL=lights_on.txt]': 0.3333333333333333}
