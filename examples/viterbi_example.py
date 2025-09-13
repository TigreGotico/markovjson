from markovjson.tokenizers import MarkovTaggerJson
from nltk.corpus import brown
from nltk import pos_tag
from pprint import pprint

# Initialize the MarkovJson model
mkov = MarkovTaggerJson(tagger=pos_tag, # can be anything that returns string tuples
                        order=1)


print("Downloading NLTK Brown Corpus...")
try:
    # Download the Brown Corpus if not already present
    brown.sents()
except LookupError:
    import nltk

    nltk.download('brown')

print("Training new model on NLTK Brown Corpus...")
# Train the model by adding sentences from the corpus
for sentence in brown.tagged_sents():
    # The add_tokens method is used for already tagged sentences
    mkov.add_tokens(sentence)


# Now, use the trained model to tag a new sentence
new_sentence = "The dogs run after the lazy cat."
print(f"\nOriginal Sentence: '{new_sentence}'")

# Use the Viterbi tagger to get the most probable sequence of tags
tagged_sentence = mkov.viterbi_tagger(new_sentence)

print("\nTagged Sentence:")
pprint(tagged_sentence)
# [('The', 'AT'),
#  ('dogs', 'NNS'),
#  ('run', 'VB'),
#  ('after', 'IN'),
#  ('the', 'AT'),
#  ('lazy', 'JJ'),
#  ('cat', 'NN'),
#  ('.', '.')]