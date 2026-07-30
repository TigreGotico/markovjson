# Topic and intent modelling

`MarkovTopic` turns the chain into a lightweight text classifier. It trains one
shared model over labelled samples and scores how strongly a document's tokens
point at each label: no separate model per class, no feature engineering.

```python
from markovjson.topic_modelling import MarkovTopic, MarkovNLPTopic
```

## How it works

Each training sample is wrapped in a label marker, e.g. `[/LABEL=lights_off]`, on
both ends before being added to the chain. Classification then asks, per label, how
much each input token drives the high-probability paths scoped to that label: the
[`calc_approximate_removal_score`](api.md#removal-scoring) heuristic. A token that
only appears under one label scores near `1.0` there and `0.0` elsewhere.

## Register topics and predict

```python
from markovjson.topic_modelling import MarkovTopic

clf = MarkovTopic(order=2)
clf.register_topic("on", ["turn on the lights", "lights on"])
clf.register_topic("off", ["turn off the lights", "lights off"])

print(clf.predict_topic("turn off"))
# {'[/LABEL=on]': 0.4, '[/LABEL=off]': 0.7}
```

`register_topic(topic_name, samples)` adds a list of strings under one label.
`predict_topic(document, thresh=0.3)` returns the labels whose mean token score
clears `thresh`. Labels come back in their internal `[/LABEL=<name>]` form.

## Train from files

One file per intent, one sample per line: the file's basename becomes the label
unless you pass `topic_name`:

```python
from markovjson.topic_modelling import MarkovTopic

clf = MarkovTopic(order=4)
clf.register_topic_from_file("intents/lights_on.txt")
clf.register_topic_from_file("intents/lights_off.txt", topic_name="off")
```

## Inspect the per-token scores

`predict_topic` averages token scores. To see them individually, use `score_tokens`
(all labels) or `score_topic` (one label):

```python
from markovjson.topic_modelling import MarkovTopic

clf = MarkovTopic(order=2)
clf.register_topic("on", ["turn on the lights", "lights on"])
clf.register_topic("off", ["turn off the lights", "lights off"])

from pprint import pprint
pprint(clf.score_tokens("turn off"))
# per-label dict of {token: score}
```

This breakdown shows *which* words pushed a prediction, handy for debugging why an
utterance landed on a label.

## API

```python
MarkovTopic(ignore_case=True, *args, **kwargs)   # *args/**kwargs pass to MarkovWordJson

register_topic(topic_name, samples) -> None
register_topic_from_file(path, topic_name=None) -> None
score_topic(topic, document, wildcards=True) -> dict[str, float]
score_tokens(document, wildcards=True) -> dict[str, dict[str, float]]
predict_topic(document, thresh=0.3, wildcards=True) -> dict[str, float]
```

- `ignore_case` lowercases samples on registration.
- `wildcards=True` (the default here) lets unseen input words still score against
  the model rather than zeroing the sequence: see
  [advanced.md](advanced.md#wildcards-for-unseen-tokens).

`MarkovNLPTopic` mixes `MarkovTopic` with `MarkovNLPJson`, so topics are learned
over POS-tagged tokens instead of bare words: reach for it when grammar, not just
vocabulary, separates your intents.

---
[← Advanced usage](advanced.md) · [Home](../README.md)
