[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/markovjson)

# MarkovJson

A Python library for creating, training, and utilizing Markov chain models with a focus on JSON-based persistence, flexible tokenization, and advanced analytical capabilities.

## Features

- Flexible Tokenization: Supports character-level (MarkovCharJson), word-level (MarkovWordJson), and advanced NLP-based tokenization with Part-of-Speech tagging (MarkovNLPJson).
- Configurable Order: Easily adjust the order of the Markov model to control the complexity of the generated sequences.
- Topic Modeling: The MarkovTopic class provides a simple yet effective way to train a model on labeled data and predict the topic of new text.
- State Removal Scoring: Calculates a heuristic score to measure the impact of a state (a token) on the overall model, useful for identifying key tokens.
- JSON Persistence: Models can be easily saved to and loaded from JSON files, making them portable and reusable.
- Reverse Modeling: Supports reverse Markov chains for applications like predicting the end of a sequence.

## Installation

`pip install markovjson`

