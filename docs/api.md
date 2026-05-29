# API reference

Public surface re-exported from `markovjson`:

```python
from markovjson import MarkovJson, MarkovCharJson, MarkovWordJson, MarkovNLPJson
from markovjson.mkov import SequenceScoringStrategy
from markovjson.topic_modelling import MarkovTopic, MarkovNLPTopic
from markovjson.nlp import normalize, pos_tag
```

`MarkovJson`, `MarkovCharJson`, `MarkovWordJson`, and `MarkovNLPJson` are on the
package root. `SequenceScoringStrategy`, the topic classes, and the NLP helpers
live in their submodules.

---

## `MarkovJson`

```python
MarkovJson(order=1, reverse=False,
           strategy=SequenceScoringStrategy.PROB_MULTIPLY)
```

The base chain. Tokenizes on whitespace by default; subclasses override
`tokenize`.

- `order` — number of previous tokens that make up a state.
- `reverse` — train a reverse chain (predict the start from the end). Swaps the
  internal meaning of the `[/START]` and `[/END]` markers.
- `strategy` — default `SequenceScoringStrategy` used by `iterate_sequences` and
  `__iter__`.

### Training

```python
add_string(text) -> None
add_tokens(tokens: list[str]) -> None
add_state(current_state: tuple, next_state: str) -> None
```

`add_string` tokenizes then calls `add_tokens`, which pads with `order` start
markers plus one end marker and increments every transition count. `add_state` is
the single-transition primitive.

### Tokenization

```python
tokenize(text, wildcards=False) -> list[str]
replace_wildcards(sequence: list[str]) -> list[str]
```

`tokenize` splits on `" "` in the base class. With `wildcards=True`, tokens not
seen in training collapse to the `[/]` wildcard marker (runs of unknowns collapse
to a single wildcard).

### Generation and sampling

```python
sample(current_state=None) -> str
generate_sequence(max_len=100, initial_state=None, pad=False, retry=2) -> list[str]
```

`sample` draws one next token weighted by transition counts, returning the
`[/NULL]` marker if the state is a dead end. `generate_sequence` repeatedly samples
until it hits the end marker or `max_len`; if it dead-ends with no `initial_state`
it restarts a fresh path up to `retry` times. The returned list still contains the
`[/START]` / `[/END]` markers — subclasses' `generate_string` strips them.

### Scoring

```python
get_state_probability(current_state: tuple, next_state: str) -> float
get_transition_weights(sequence, wildcards=False) -> tuple[list[int], list[float]]
get_sequence_prob(sequence,
                  strategy=SequenceScoringStrategy.PROB_MULTIPLY,
                  wildcards=False) -> float
```

`get_state_probability` is one transition's `count / total`. `get_transition_weights`
returns `(raw_counts, probabilities)` for every step in a sequence.
`get_sequence_prob` reduces those into a single number per the chosen strategy;
it returns `0` for an impossible sequence.

```python
from markovjson import MarkovWordJson
w = MarkovWordJson(order=1)
w.add_string("turn on the lights")
w.add_string("turn off the lights")
print(w.get_sequence_prob("turn on the lights"))   # 0.5
```

### Enumerating paths

```python
iterate_sequences(initial_state=None, max_len=10, pad=False, strategy=None,
                  max_depth=25, thresh=0.01) -> Iterator[tuple[list[str], float]]
__iter__()   # iterate_sequences from the start state
```

Yields `(sequence, score)` pairs for complete paths whose score clears `thresh`.
`max_depth` bounds the recursion; `max_len` bounds path length.

### State helpers

```python
get_state(initial_state=None, pad=True) -> tuple
state2sequence(initial_state, pad=False, wildcards=False) -> list[str]
sequence2states(sequence, wildcards=False) -> list[tuple]
```

`tokens` (property) returns the unique token vocabulary seen in training.

### Persistence

```python
save(filename) -> None
load(filename) -> MarkovJson   # returns self
```

`save` serializes `order`, the sequence markers, `reverse_modelling`, and the
transition table to JSON via `json_database`. `load` mutates the instance in place
and returns it, so `MarkovWordJson(order=1).load("m.json")` works.

### Removal scoring

```python
calc_approximate_removal_score(state, sequences=None, required_states=None,
                               blacklisted_states=None, max_seqs=100,
                               *args, **kwargs) -> float
```

A heuristic in `[0.0, 1.0]` for how much a single token (`state`) drives the
high-probability paths of the model, optionally scoped to paths that must contain
`required_states` and must not contain `blacklisted_states`. This is the engine
behind [topic_modelling.md](topic_modelling.md).

### `SequenceScoringStrategy`

A `str` `Enum` of reducers for `get_sequence_prob`. Raw-count variants
(`MAX`, `MIN`, `SUM`, `MULTIPLY`, `AVERAGE`) operate on integer transition counts;
`PROB_*` variants operate on per-step probabilities:

```python
from markovjson.mkov import SequenceScoringStrategy
SequenceScoringStrategy.PROB_MULTIPLY   # default
# PROB_AVERAGE, PROB_MAX, PROB_MIN, PROB_SUM
# MAX, MIN, SUM, MULTIPLY, AVERAGE
```

---

## `MarkovCharJson(MarkovJson)`

Character-level chain. `tokenize` splits into individual characters.

```python
generate_string(*args, **kwargs) -> str
```

Generates via `generate_sequence` and concatenates the characters (markers
removed). Accepts the same kwargs as `generate_sequence` (`max_len`,
`initial_state`, `pad`, `retry`).

```python
from markovjson import MarkovCharJson
m = MarkovCharJson(order=2)
for name in ["alice", "alma", "alva", "alan"]:
    m.add_string(name)
print(m.generate_string(max_len=10))
```

---

## `MarkovWordJson(MarkovJson)`

Word-level chain (inherits the whitespace `tokenize`).

```python
generate_string(*args, **kwargs) -> str   # joins tokens with " "
```

---

## `MarkovNLPJson(MarkovWordJson)`

POS-tagged chain. Tokens are `word [/TAG=POS]` strings produced by NLTK's
`pos_tag`; the model also tracks per-tag emission counts so it can tag new text.

```python
MarkovNLPJson(normalize=False, wildcard_postags=None, *args, **kwargs)
```

- `normalize` — lemmatize/clean text (via `markovjson.nlp.normalize`) before tagging.
- `wildcard_postags` — list of POS tags to treat as wildcards.

```python
viterbi_tagger(sentence, unknown_word_prob=1e-10) -> list[tuple[str, str]]
generate_string(*args, **kwargs) -> str                 # words only, tags dropped
generate_tagged_string(*args, **kwargs) -> list[tuple[str, str]]
```

`viterbi_tagger` returns the most likely `(word, tag)` sequence for a sentence
using the trained transition + emission probabilities. `generate_tagged_string`
returns generated output as `(word, tag)` tuples.

`MarkovNLPJson` subclasses `MarkovTaggerJson` (in `markovjson.tokenizers`), which
is the generic version taking any `tagger` callable. `MarkovNLPJson` wires in the
NLTK tagger for you.

---

## NLP helpers (`markovjson.nlp`)

```python
normalize(X, stemmer=None, lemmatize=True) -> list[str]
pos_tag(sentence) -> list[tuple[str, str]]
```

`normalize` strips special characters, lowercases, and lemmatizes (WordNet by
default) — it always returns a *list* of cleaned documents, even for a single
string input. `pos_tag` wraps NLTK tokenize + tag and self-downloads the required
corpora on first use.

## Where next

- [quickstart.md](quickstart.md) — install and the core idea
- [advanced.md](advanced.md) — strategies, reverse models, wildcards, gotchas
- [topic_modelling.md](topic_modelling.md) — `MarkovTopic` for intent classification
