[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/markovjson)

# MarkovJson

MarkovJson is a Python library for Markov chain models over text. It builds
order-N chains from tokens, saves and loads them as JSON, and supports
character-level, word-level, and POS-tagged tokenization. On top of the chain,
it adds sequence scoring, topic and intent classification, and a heuristic
score for how much a single token drives a model's high-probability paths.

## Features

- Flexible tokenization: character-level (`MarkovCharJson`), word-level
  (`MarkovWordJson`), and NLP tokenization with part-of-speech tagging
  (`MarkovNLPJson`).
- Configurable order: set how many previous tokens make up a state, to
  control the complexity of generated sequences.
- Topic modeling: `MarkovTopic` trains one shared model on labeled samples
  and predicts the topic of new text.
- Removal scoring: `calc_approximate_removal_score` measures how much a
  single state (token) drives a model, useful for finding key tokens.
- JSON persistence: save a model to a JSON file and load it back, so trained
  models are portable.
- Reverse modeling: train a reverse chain to predict the start of a sequence
  from its end.

## Install

```bash
pip install markovjson
```

This pulls in [json_database](https://github.com/TigreGotico/json_database)
for JSON persistence, and `nltk` for POS tagging and lemmatization. NLTK
corpora (`punkt`, `averaged_perceptron_tagger`, `wordnet`, `stopwords`)
download on first use of the NLP path.

## Usage

Train a character-level model on a few names and generate a new one:

```python
from markovjson import MarkovCharJson

m = MarkovCharJson(order=2)
for name in ["alice", "alma", "alva", "alan"]:
    m.add_string(name)

print(m.generate_string(max_len=10))   # e.g. "alan"
```

Score how likely an existing sequence is:

```python
from markovjson import MarkovWordJson

w = MarkovWordJson(order=1)
w.add_string("turn on the lights")
w.add_string("turn off the lights")

print(w.get_sequence_prob("turn on the lights"))   # 0.5
```

See the docs for more:

- [Quickstart](docs/quickstart.md): install and the core idea
- [API reference](docs/api.md): every public class, method, kwarg, and return shape
- [Advanced usage](docs/advanced.md): scoring strategies, reverse models, wildcards, gotchas
- [Topic and intent modelling](docs/topic_modelling.md): intent and topic classification on the chain

## Related projects

- [json_database](https://github.com/TigreGotico/json_database): the JSON
  persistence layer MarkovJson uses to save and load models.
- [ovos-markov-chat-plugin](https://github.com/TigreGotico/ovos-markov-chat-plugin):
  Markov chain chat agent for OVOS personas, built on this library.
- [ovos-markov-pipeline-plugin](https://github.com/TigreGotico/ovos-markov-pipeline-plugin):
  OVOS intent pipeline plugin using Markov chain perplexity ensemble.

## License

Apache License 2.0.
