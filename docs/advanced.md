# Advanced usage

Recipes and sharp edges once you are past the [quickstart](quickstart.md).

## Choosing a scoring strategy

`get_sequence_prob` and `iterate_sequences` reduce a sequence's per-step weights
into one number. The strategy decides how. `PROB_*` variants use probabilities
(0-1 per step). The raw variants use integer transition counts.

```python
from markovjson import MarkovWordJson
from markovjson.mkov import SequenceScoringStrategy as S

w = MarkovWordJson(order=1)
w.add_string("turn on the lights")
w.add_string("turn off the lights")

phrase = "turn on the lights"
print(w.get_sequence_prob(phrase, strategy=S.PROB_MULTIPLY))   # product of step probs
print(w.get_sequence_prob(phrase, strategy=S.PROB_AVERAGE))    # mean step prob
print(w.get_sequence_prob(phrase, strategy=S.MIN))             # rarest raw transition
```

Rules of thumb:

- `PROB_MULTIPLY` (default) punishes any single weak link: good for "is this whole
  phrase well-formed?"
- `PROB_AVERAGE` is forgiving of one odd step: good for fuzzy matching.
- `MIN` / `MAX` on raw counts surface the rarest / most common single transition.

## Setting the order

`order` is the context window. Higher order means more faithful reproduction of the
training data but sparser tables and less novelty.

```python
from markovjson import MarkovCharJson

corpus = ["alice", "alma", "alva", "alan", "alba"]
for order in (1, 2):
    m = MarkovCharJson(order=order)
    for name in corpus:
        m.add_string(name)
    print(order, m.generate_string(max_len=12))
```

Order 1 wanders. Order 2 stays closer to the source spellings.

## Reverse models

Pass `reverse=True` to predict backwards: useful for completing the *start* of a
sequence, or generating endings. The chain swaps the internal roles of the start
and end markers. `generate_string` flips the output back to reading order for you.

```python
from markovjson import MarkovWordJson

rev = MarkovWordJson(order=1, reverse=True)
rev.add_string("please turn off the lights")
rev.add_string("now turn off the lights")
print(rev.generate_string(max_len=10))
```

## Wildcards for unseen tokens

When scoring real-world input, tokens absent from training would make every
sequence impossible. `wildcards=True` rewrites unknown tokens to the `[/]` marker
so a partial match still scores:

```python
from markovjson import MarkovWordJson

w = MarkovWordJson(order=1)
w.add_string("turn on the lights")

print(w.get_sequence_prob("please turn on the lights", wildcards=True))
```

Consecutive unknowns collapse to a single wildcard, so noise does not explode the
state space.

## Enumerating likely completions

`iterate_sequences` yields complete `(sequence, score)` paths above `thresh`. Use
it to list what a model considers plausible:

```python
from markovjson import MarkovWordJson

w = MarkovWordJson(order=1)
w.add_string("turn on the lights")
w.add_string("turn off the lights")

for seq, score in w.iterate_sequences(thresh=0.1):
    words = [t for t in seq if not t.startswith("[/")]
    print(round(score, 3), " ".join(words))
```

## POS tagging with Viterbi

`MarkovNLPJson` learns transition *and* emission counts, so a trained model can tag
new sentences with the Viterbi algorithm:

```python
from markovjson import MarkovNLPJson

m = MarkovNLPJson(order=2)
m.add_string("the cat sat on the mat")
m.add_string("the dog ran on the road")

print(m.viterbi_tagger("the cat ran"))   # [(word, tag), ...]
```

Unknown words fall back to `unknown_word_prob`. If the model has no tags at all,
every token comes back tagged `"UNK"`.

## Gotchas

- **Tuple-key round-trip.** `save` stringifies the tuple state keys. `load` rebuilds
  them by slicing those strings. Tokens containing `', '` can corrupt on reload.
  Keep tokens free of that literal substring, or persist in a context where you
  control the vocabulary.
- **Tagged tokens are single-word.** `MarkovTaggerJson` stores `"<word> [/TAG=<tag>]"`
  and `generate_tagged_string` splits on the last space. Multi-word tokens are not
  supported on that path.
- **`generate_sequence` returns markers.** The raw list includes `[/START]` /
  `[/END]`. Filter them yourself, or use a subclass's `generate_string`.
- **`normalize` returns a list.** Even given a single string,
  `markovjson.nlp.normalize` returns a list of documents: index `[0]` for one.

---
[← API reference](api.md) · [Home](../README.md) · [Topic and intent modelling →](topic_modelling.md)
