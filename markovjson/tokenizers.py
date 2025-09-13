import math
from collections import defaultdict
from typing import List, Tuple, Any, Callable

from markovjson.mkov import MarkovJson
from markovjson.nlp import pos_tag


class MarkovCharJson(MarkovJson):
    """
    A Markov chain model specifically designed for character-level tokenization.
    """

    def tokenize(self, text: str, wildcards: bool = False) -> List[str]:
        """
        Tokenizes a given string into a list of individual characters.

        Args:
            text (str): The input string to tokenize.
            wildcards (bool, optional): Whether to replace unseen characters with a wildcard token.
                                         Defaults to False.

        Returns:
            List[str]: A list of characters from the input string.
        """
        sequence = list(text)
        if wildcards:
            sequence = self.replace_wildcards(sequence)
        if self.reverse_modelling:
            sequence.reverse()
        return sequence

    def generate_string(self, *args: Any, **kwargs: Any) -> str:
        """
        Generates a string by concatenating the characters from a generated sequence.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: A single string generated from the Markov chain.
        """
        seq = self.generate_sequence(*args, **kwargs)
        s = "".join([s for s in seq if
                     s != self.START_OF_SEQ and s != self.END_OF_SEQ])
        if self.reverse_modelling:
            return s[::-1]
        return s


class MarkovWordJson(MarkovJson):
    """
    A Markov chain model for word-level tokenization.
    """

    def generate_string(self, *args: Any, **kwargs: Any) -> str:
        """
        Generates a string by joining the words from a generated sequence with spaces.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: A single string generated from the Markov chain.
        """
        seq = self.generate_sequence(*args, **kwargs)
        s = " ".join([s for s in seq if
                      s != self.START_OF_SEQ and s != self.END_OF_SEQ])
        if self.reverse_modelling:
            return s[::-1]
        return s


class MarkovTaggerJson(MarkovWordJson):
    """
    This class extends the `MarkovWordJson` to handle tokens that are
    word-tag pairs, allowing the model to learn grammar and syntax.
    """

    def __init__(self,
                 tagger: Callable[[str], List[Tuple[str, str]] | List[str]],
                 normalize: bool = False, wildcard_postags: List[str] | None = None,
                 *args: Any, **kwargs: Any) -> None:
        """
        Initializes the MarkovNLPJson model.

        Args:
            normalize (bool, optional): Whether to normalize text before tokenization.
                                        Defaults to False.
            wildcard_postags (List[str] | None, optional): A list of POS tags to
                                                         treat as wildcards. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        self.wildcard_postags = wildcard_postags or []
        self.emissions = defaultdict(lambda: defaultdict(int))
        self.tag_counts = defaultdict(int)
        self.tagger = tagger

    def tokenize(self, text: str,
                 wildcards: bool = False) -> List[Tuple[str, str]] | List[str]:
        """
        Tokenizes a sentence and returns a list of (word, tag) tuples if postag is enabled.
        Otherwise, falls back to the parent class's word tokenization.

        Args:
            text (str): The input text to tokenize.
            wildcards (bool, optional): Whether to use wildcard tokens. Defaults to False.

        Returns:
            List[Tuple[str, str]] | List[str]: A list of word-tag tuples or
                                             a list of words depending on `postag`.
        """
        if self.normalize:
            try:
                from markovjson.nlp import normalize
                text = normalize(text)
            except ImportError:
                pass
        # return a list of (word, tag) tuples
        return self.tagger(text)

    def add_tokens(self, tokens: List[Tuple[str, str]]) -> None:
        """
        Adds a sequence of (word, tag) tokens to the Markov chain.

        Args:
            tokens (List[Tuple[str, str]]): A list of (word, tag) tuples to add to the model.
        """
        # Update emission counts for each (word, tag) pair
        for word, tag in tokens:
            self.emissions[tag][word] += 1
            self.tag_counts[tag] += 1

        # Tokenize and create sequences
        tokenized_sequence = [f"{word} [/TAG={tag}]" for word, tag in tokens]

        # Now, add the tags themselves to the Markov chain
        tokenized_sequence = [self.START_OF_SEQ] * self.order + tokenized_sequence + [self.END_OF_SEQ]
        if self.reverse_modelling:
            tokenized_sequence.reverse()
        for i in range(len(tokenized_sequence) - self.order):
            key = tuple(tokenized_sequence[i:i + self.order])
            if key not in self.records:
                self.records[key] = {}
            if tokenized_sequence[i + self.order] not in self.records[key]:
                self.records[key][tokenized_sequence[i + self.order]] = 0
            self.records[key][tokenized_sequence[i + self.order]] += 1

    def viterbi_tagger(self, sentence: str, unknown_word_prob: float = 1e-10) -> List[Tuple[str, str]]:
        """
        Tags a sentence using the Viterbi algorithm for any model order > 0.

        The Viterbi algorithm finds the most likely sequence of hidden states
        (POS tags) given a sequence of observations (words). It uses a dynamic
        programming approach to efficiently search through all possible tag sequences.

        Args:
            sentence (str): The sentence to tag.
            unknown_word_prob (float): A small probability for words not seen
                                       during training to prevent zero probabilities.

        Returns:
            List[Tuple[str, str]]: A list of (word, tag) tuples representing the most likely sequence.
        """
        # Step 1: Pre-computation and Initialization
        words = [w[0] for w in pos_tag(sentence)]
        all_tags = list(self.tag_counts.keys())
        if not all_tags:
            return [(word, "UNK") for word in words]

        # Viterbi tables: stores the max log-probability for a path ending in a specific state
        viterbi = defaultdict(lambda: defaultdict(lambda: -math.inf))
        # Backpointer table: stores the previous state tuple that led to the max probability
        backpointer = defaultdict(lambda: defaultdict(lambda: None))

        # we use `self.START_OF_SEQ` to represent the initial state context
        initial_tokens = [self.START_OF_SEQ] * self.order

        # Initialize probabilities for the first word
        first_word = words[0]
        for tag in all_tags:
            # Construct the full state tuple for the first word, including start padding
            current_state = tuple(initial_tokens[:-1] + [f"{first_word} [/TAG={tag}]"])

            # Transition probability from the start state to the current state
            transition_prob = self.get_state_probability(tuple(initial_tokens), current_state[-1])

            # Emission probability of the word given the tag
            emission_prob = self.emissions[tag].get(first_word, 0) / self.tag_counts[tag]

            # Use a small value for zero probabilities to avoid `math domain error`
            if transition_prob == 0:
                transition_prob = unknown_word_prob
            if emission_prob == 0:
                emission_prob = unknown_word_prob

            viterbi[0][current_state] = math.log(transition_prob) + math.log(emission_prob)

        # Step 2: Viterbi Trellis Recursion
        # Loop through the rest of the words in the sentence
        for i in range(1, len(words)):
            current_word = words[i]
            # Iterate through all possible previous states from the previous word
            for prev_state in viterbi[i - 1].keys():
                prev_prob = viterbi[i - 1][prev_state]

                # Iterate through all possible tags for the current word
                for tag in all_tags:
                    # Construct the next state tuple by shifting the window
                    next_state = tuple(prev_state[1:] + (f"{current_word} [/TAG={tag}]",))

                    # Get the transition probability
                    transition_prob = self.get_state_probability(prev_state, next_state[-1])

                    # Get the emission probability
                    emission_prob = self.emissions[tag].get(current_word, 0) / self.tag_counts[tag]

                    if transition_prob == 0:
                        transition_prob = unknown_word_prob

                    if emission_prob == 0:
                        emission_prob = unknown_word_prob

                    # Calculate the probability of the current path
                    current_path_prob = prev_prob + math.log(transition_prob) + math.log(emission_prob)

                    # If this path is better than any existing path to the current state, update the tables
                    if current_path_prob > viterbi[i][next_state]:
                        viterbi[i][next_state] = current_path_prob
                        backpointer[i][next_state] = prev_state

        # Step 3: Back-pointer Tracking
        best_path_prob = -math.inf
        last_state = None

        # Find the best final state and path probability by transitioning to END_OF_SEQ
        final_word_index = len(words) - 1
        if final_word_index not in viterbi:
            return [(word, "UNK") for word in words]

        for state, prob in viterbi[final_word_index].items():
            transition_prob = self.get_state_probability(state, self.END_OF_SEQ)
            if transition_prob == 0:
                transition_prob = unknown_word_prob

            final_prob = prob + math.log(transition_prob)
            if final_prob > best_path_prob:
                best_path_prob = final_prob
                last_state = state

        # Reconstruct the path by traversing the backpointers
        if last_state is None:
            return [(word, "UNK") for word in words]

        tagged_sequence = []
        current_state = last_state
        for i in range(final_word_index, -1, -1):
            tag = current_state[-1].split(" [/TAG=")[-1][:-1]
            word = words[i]
            tagged_sequence.append((word, tag))
            if i > 0:
                current_state = backpointer[i][current_state]

        tagged_sequence.reverse()
        return tagged_sequence

    def generate_string(self, *args: Any, **kwargs: Any) -> str:
        """
        Generates a string, removing the special POS tags from the output.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The generated text without the tags.
        """
        seq = self.generate_sequence(*args, **kwargs)
        s = " ".join([s.split(" ")[0] for s in seq if
                      s != self.START_OF_SEQ and s != self.END_OF_SEQ])
        if self.reverse_modelling:
            return s[::-1]
        return s

    def generate_tagged_string(self, *args: Any, **kwargs: Any) -> List[Tuple[str, str]]:
        """
        Generates a sequence and returns it as a list of (word, tag) tuples.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Tuple[str, str]]: The generated sequence as a list of (word, tag) tuples.
        """
        seq = self.generate_sequence(*args, **kwargs)
        tagged_list = []
        for token in seq:
            if token not in [self.START_OF_SEQ, self.END_OF_SEQ]:
                parts = token.rsplit(" ", 1)
                word = parts[0]
                tag = parts[1].replace("[/TAG=", "").replace("]", "")
                tagged_list.append((word, tag))
        return tagged_list


class MarkovNLPJson(MarkovTaggerJson):
    """
    A Markov chain model for advanced Natural Language Processing (NLP) tasks,
    including Part-of-Speech (POS) tagging.

    This class extends the `MarkovTaggerJson` to use NLTK pos_tag
    """

    def __init__(self,
                 normalize: bool = False, wildcard_postags: List[str] | None = None,
                 *args: Any, **kwargs: Any) -> None:
        from markovjson.nlp import pos_tag
        super().__init__(pos_tag, normalize, wildcard_postags, *args, **kwargs)
