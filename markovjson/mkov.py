import random
from json_database import JsonStorage
from enum import Enum


class SequenceScoringStrategy(str, Enum):
    MAX = "max"
    MIN = "min"
    MULTIPLY = "multiply"
    AVERAGE = "average"
    SUM = "sum"
    PROB_AVERAGE = "prob_average"
    PROB_MAX = "prob_max"
    PROB_MIN = "prob_min"
    PROB_SUM = "prob_sum"
    PROB_MULTIPLY = "prob_multiply"


class MarkovJson:
    def __init__(self, order=1, reverse=False,
                 strategy=SequenceScoringStrategy.PROB_MULTIPLY):
        self.START_OF_SEQ = "[/START]"
        self.END_OF_SEQ = "[/END]"
        self.NULL_SEQ = "[/NULL]"
        self.WILDCARD_SEQ = "[/]"
        self.reverse_modelling = reverse
        if not reverse:
            self.START_OF_SEQ = "[/START]"
            self.END_OF_SEQ = "[/END]"
        else:
            self.START_OF_SEQ = "[/END]"
            self.END_OF_SEQ = "[/START]"
        self.order = order
        self.records = {}
        self.strategy = strategy
        self._current_state = [self.START_OF_SEQ] * self.order

    # tokenization
    @property
    def tokens(self):
        toks = []
        for state in self.records:
            toks += list(state)
        return list(set(toks))

    def tokenize(self, text, wildcards=False):
        sequence = text.split(" ")
        if wildcards:
            sequence = self.replace_wildcards(sequence)
        if self.reverse_modelling:
            sequence.reverse()
        return sequence

    def replace_wildcards(self, sequence):
        # replace unknown tokens with wildcard
        prev_token = self.START_OF_SEQ
        for idx, s in enumerate(sequence):
            if s not in self.tokens:
                if prev_token == self.WILDCARD_SEQ:
                    sequence[idx] = None
                    continue
                else:
                    sequence[idx] = self.WILDCARD_SEQ
            prev_token = sequence[idx]
        sequence = [s for s in sequence if s is not None]
        return sequence

    # "training"
    def add_string(self, text):
        tokens = self.tokenize(text)
        self.add_tokens(tokens)

    def add_tokens(self, tokens):
        tokens = [self.START_OF_SEQ] * self.order + tokens + [self.END_OF_SEQ]

        for i in range(len(tokens) - self.order):
            current_state = tuple(tokens[i:i + self.order])
            next_state = tokens[i + self.order]
            self.add_state(current_state, next_state)

    def add_state(self, current_state, next_state):
        if current_state not in self.records:
            self.records[current_state] = {}

        if next_state not in self.records[current_state]:
            self.records[current_state][next_state] = 0

        self.records[current_state][next_state] += 1

    # sequence handling
    def get_state(self, initial_state=None, pad=True):
        sequence = self.state2sequence(initial_state, pad=pad)
        return tuple(sequence[-self.order:])

    def state2sequence(self, initial_state, pad=False, wildcards=False):
        if initial_state is None:
            sequence = [self.START_OF_SEQ] * self.order
        elif isinstance(initial_state, str):
            sequence = self.tokenize(initial_state, wildcards=wildcards)
        else:
            sequence = initial_state[:]

        if pad or len(sequence) < self.order:
            sequence = [self.START_OF_SEQ] * self.order + sequence
        sequence = [s for s in sequence if s]  # filter empty strings and such
        return sequence

    def sequence2states(self, sequence, wildcards=False):
        if isinstance(sequence, str):
            sequence = self.state2sequence(sequence, wildcards=wildcards)
        # convert a list of tokens into tuples of states
        # that can be looked up in self.records
        states = []
        for i in range(len(sequence)):
            state = tuple(sequence[i:self.order + i])
            if len(state) < self.order:
                break
            states.append(state)
        return states

    def get_transition_weights(self, sequence, wildcards=False):
        states = self.sequence2states(sequence, wildcards=wildcards)
        weights = []  # raw integer count
        avg_weights = []  # probs 0 to 1 for each transition
        for idx, state in enumerate(states):

            next_state = states[idx + 1] if idx < len(states) - 1 else None
            if not next_state:
                continue
            jump = next_state[-1]
            # transition does not exist in the model
            if state not in self.records or not self.records[state].get(jump):
                w = 0
                t = 1
            else:
                w = self.records[state][jump]
                t = sum(self.records[state][s] for s in self.records[state])
            weights.append(w)
            avg_weights.append(w / t)
        return weights, avg_weights

    def get_sequence_prob(self, sequence,
                          strategy=SequenceScoringStrategy.PROB_MULTIPLY,
                          wildcards=False):
        weights, avg_weights = self.get_transition_weights(sequence,
                                                           wildcards=wildcards)
        if not weights:  # sequence is not possible
            return 0

        if strategy == SequenceScoringStrategy.AVERAGE:
            return sum(weights) / len(weights)

        if strategy == SequenceScoringStrategy.PROB_AVERAGE:
            return sum(avg_weights) / len(avg_weights)

        if strategy == SequenceScoringStrategy.MAX:
            return max(weights)

        if strategy == SequenceScoringStrategy.PROB_MAX:
            return max(avg_weights)

        if strategy == SequenceScoringStrategy.MIN:
            return min(weights)

        if strategy == SequenceScoringStrategy.PROB_MIN:
            return min(avg_weights)

        if strategy == SequenceScoringStrategy.SUM:
            return sum(weights)

        if strategy == SequenceScoringStrategy.PROB_SUM:
            return sum(avg_weights)

        if strategy == SequenceScoringStrategy.MULTIPLY:
            score = 1
            for c in weights:
                score = score * c
            return score

        if strategy == SequenceScoringStrategy.PROB_MULTIPLY:
            score = 1
            for c in avg_weights:
                score = score * c
            return score

    def iterate_sequences(self, initial_state=None, max_len=10, pad=False,
                          strategy=None, max_depth=25, thresh=0.01):
        # TODO max_loops, how many times can end up in same state before
        #  path starts being ignored
        strategy = strategy or self.strategy
        current_state = self.get_state(initial_state, pad)
        sequence = self.state2sequence(initial_state, pad)

        if current_state not in self.records:
            return

        for possible_next, val in self.records[current_state].items():

            max_depth -= 1
            if max_depth <= 0:
                return

            seq = sequence + [possible_next]

            if len(seq) >= max_len + self.order:
                # sequence is too big
                # accounts for some infinite loops that can happen
                # without max_len, like autocorrect loops on old phones
                return

            # found a full path!
            if possible_next == self.END_OF_SEQ:
                score = self.get_sequence_prob(seq, strategy)
                if score >= thresh:
                    yield seq, score

            # check all sequences starting from the new sequence
            else:
                for seq2, conf2 in self.iterate_sequences(
                        seq, max_len=max_len, strategy=strategy,
                        thresh=thresh, max_depth=max_depth):
                    yield seq2, conf2

    def __iter__(self):
        for p in self.iterate_sequences(self._current_state):
            yield p

    # sampling
    def sample(self, current_state=None):
        if current_state is None:
            current_state = tuple([self.START_OF_SEQ] * self.order)
        elif isinstance(current_state, str):
            sequence = [self.START_OF_SEQ] * self.order + \
                       self.tokenize(current_state)
            current_state = tuple(sequence[-self.order:])

        possible_next = self.records.get(current_state)
        if not possible_next:
            return self.NULL_SEQ

        n = sum(possible_next.values())

        m = random.randint(0, n)
        count = 0
        for k, v in possible_next.items():
            count += v
            if m <= count:
                return k

    def generate_sequence(self, max_len=100, initial_state=None, pad=False, retry=2):
        sequence = self.state2sequence(initial_state, pad=pad)
        for i in range(max_len):
            current_state = tuple(sequence[-self.order:])
            next_token = self.sample(current_state)
            if next_token == self.NULL_SEQ:
                if not initial_state and retry > 0: # find a new valid path
                    return self.generate_sequence(max_len=max_len, pad=pad, retry=retry - 1)
                continue
            sequence.append(next_token)
            if next_token == self.END_OF_SEQ:
                return sequence
        return sequence

    # persistence
    def save(self, filename):
        """
        Saves Markov chain to filename
        :param filename: string - where to save chain
        :return: None
        """
        with JsonStorage(filename) as db:
            db["order"] = self.order
            db["START_OF_SEQ"] = self.START_OF_SEQ
            db["END_OF_SEQ"] = self.END_OF_SEQ
            db["reverse_modelling"] = self.reverse_modelling
            # convert tuple keys to strings
            db["records"] = {str(k): v for k, v in self.records.items()}

    def load(self, filename):
        """
        Saves Markov chain to filename
        :param filename: string - where to save chain
        :return: None
        """
        with JsonStorage(filename) as db:
            if self._current_state == [self.START_OF_SEQ] * self.order:
                self._current_state = [db["START_OF_SEQ"]] * db["order"]
            self.order = db["order"]
            self.START_OF_SEQ = db["START_OF_SEQ"]
            self.END_OF_SEQ = db["END_OF_SEQ"]
            self.reverse_modelling = db.get("reverse_modelling") or False
            # convert str keys back to tuples
            self.records = {
                tuple(k.replace("',)", "')")[2:-2].split("', '")): v
                for k, v in db["records"].items()}
        return self

    # metrics
    def calc_approximate_removal_score(self, state, sequences=None,
                                       required_states=None,
                                       blacklisted_states=None, max_seqs=100,
                                       *args, **kwargs):
        """
        Calculates a heuristic score for a token's importance within the model,
        specifically tailored for tasks like topic or intent classification.

        This score approximates how much a given token ('state') contributes to the
        high-probability sequences within a specific context (e.g., a topic). A
        higher score (closer to 1.0) means the token is a strong indicator for that
        context.

        The method uses several opinionated heuristics instead of a pure
        probabilistic approach, making it effective for sparse datasets:

        1.  **Biased Sampling**: It deliberately over-samples sequences containing the
            token to ensure it can be scored, even if it's rare. This trades
            probabilistic purity for practical relevance.
        2.  **Max-Probability Normalization**: Probabilities are scaled relative to the
            *most likely* sequence in the sample. This acts as a contrast enhancement,
            focusing the score on strong, defining patterns rather than the entire
            distribution.

        Args:
            state (str): The token (e.g., a word) to be scored.
            sequences (list, optional): A pre-computed list of sequences to analyze.
                                        If None, sequences will be generated.
            required_states (list, optional): A list of tokens that MUST be present
                                              in a sequence for it to be included
                                              in the analysis. Used to scope the
                                              analysis to a specific topic.
            blacklisted_states (list, optional): A list of tokens that MUST NOT be
                                                 present in any analyzed sequence.
            max_seqs (int): The maximum number of sequences to sample.

        Returns:
            float: A score between 0.0 and 1.0 representing the token's
                   heuristic importance for the given context.
        """
        if state == self.WILDCARD_SEQ:
            return 0
        required_states = required_states or [self.END_OF_SEQ]
        blacklisted_states = blacklisted_states or [self.NULL_SEQ]

        # --- Step 1: Biased Sampling ---
        # If no sequences are provided, generate a sample set. This sampling is
        # intentionally biased to ensure the token being scored is represented,
        # which is crucial for sparse data where the token might otherwise be missed.
        if not sequences:
            sequences = []
            # First half of the sample starts with the state, guaranteeing its presence.
            for p in self.iterate_sequences(
                    thresh=0.01,
                    strategy=SequenceScoringStrategy.PROB_MULTIPLY,
                    initial_state=state, *args, **kwargs):
                sequences.append(p)
                if len(sequences) >= max_seqs / 2:
                    break
            # Second half is sampled randomly to represent the general model.
            for p in self.iterate_sequences(
                    thresh=0.01,
                    strategy=SequenceScoringStrategy.PROB_MULTIPLY,
                    *args, **kwargs):
                sequences.append(p)
                if len(sequences) >= max_seqs:
                    break

        # --- Step 2: Filtering ---
        # Filter the sample to match the desired context (e.g., a specific topic).
        if required_states:
            sequences = [p for p in sequences
                         if all([t in p[0] for t in required_states])]
        if blacklisted_states:
            sequences = [p for p in sequences
                         if not any([t in p[0] for t in blacklisted_states])]

        if not sequences:
            return 0  # No relevant paths found in the model.

        # Partition the filtered sequences into two groups.
        with_state = [p for p in sequences if state in p[0]]

        # --- Step 3: Heuristic Scoring ---
        # This scoring method is designed to measure relative importance.

        # Find the maximum probability in the sample. This will be our baseline (1.0)
        # for normalization. This enhances contrast by measuring everything relative
        # to the "strongest signal" or most prototypical example.
        max_p = max(p[1] for p in sequences)
        if max_p == 0:
            return 0

        # Calculate the total "weight" of all sequences, normalized by the max probability.
        # This is not a true probability sum, but a heuristic measure of the total
        # importance of all sequences in the sample.
        total_weight = sum(p[1] / max_p for p in sequences)
        if total_weight == 0:
            return 0

        # Calculate the weight of only the sequences that contain the target state.
        weight_with_state = sum(p[1] / max_p for p in with_state)

        # --- Step 4: Final Score Calculation ---
        # The score is the proportion of the total heuristic weight that is associated
        # with the sequences containing the state.
        # This directly answers: "Of the important patterns for this topic, how much
        # of that importance is tied to this specific word?"
        score = weight_with_state / total_weight
        return score