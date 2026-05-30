# Quickstart — markovjson in five minutes

`markovjson` builds order-N Markov chains over tokens and persists them as plain
JSON. You pick how text becomes tokens — characters, words, or POS-tagged words —
and the same chain machinery generates new sequences, scores existing ones, and
saves to disk.

## 1. Install

```bash
pip install markovjson
```

Pulls in `json_database` (JSON persistence) and `nltk` (POS tagging,
lemmatization). NLTK corpora (`punkt`, `averaged_perceptron_tagger`, `wordnet`,
`stopwords`) download themselves the first time you use the NLP path.

## 2. The one idea

A model keeps a table of *states* → *next-token counts*. A state is a tuple of the
last `order` tokens. Training just walks your text and increments those counts;
generation walks the table back the other way, sampling the next token by weight.

Every model is a `MarkovJson` underneath. The subclass you choose only decides how
a string splits into tokens and how a sequence joins back into a string:

| Class | Token unit | Joins with |
| --- | --- | --- |
| `MarkovCharJson` | one character | `""` |
| `MarkovWordJson` | one space-split word | `" "` |
| `MarkovNLPJson` | a `word [/TAG=POS]` pair | `" "` (tag stripped) |

## 3. First real model

Train a character model on a handful of names and sample a new one:

```python
from markovjson import MarkovCharJson

m = MarkovCharJson(order=2)
for name in ["alice", "alma", "alva", "alan"]:
    m.add_string(name)

print(m.generate_string(max_len=10))   # e.g. "alan" — a plausible new name
```

`add_string` tokenizes and trains. `generate_string` samples a fresh sequence and
joins it for you, dropping the internal `[/START]` / `[/END]` markers.

## 4. Score instead of generate

The same chain tells you how likely an existing sequence is. Train two intents and
ask which transitions a phrase took:

```python
from markovjson import MarkovWordJson

w = MarkovWordJson(order=1)
w.add_string("turn on the lights")
w.add_string("turn off the lights")

print(w.get_sequence_prob("turn on the lights"))   # 0.5
```

The `0.5` is the product of per-transition probabilities: `turn` splits evenly
between `on` and `off`, everything else is deterministic.

## 5. Save and reload

Models round-trip through JSON, so a trained chain is a portable artifact:

```python
from markovjson import MarkovWordJson

w = MarkovWordJson(order=1)
w.add_string("turn on the lights")
w.save("lights.json")

reloaded = MarkovWordJson(order=1).load("lights.json")
print(reloaded.records)
```

`save` writes the order, the sequence markers, and the transition table. `load`
returns `self`, so you can chain it onto a constructor as above.

## Where next

- [api.md](api.md) — every public class, method, kwarg, and return shape
- [advanced.md](advanced.md) — scoring strategies, reverse models, wildcards, gotchas
- [topic_modelling.md](topic_modelling.md) — intent/topic classification on top of the chain
