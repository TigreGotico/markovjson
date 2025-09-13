from markovjson import MarkovNLPJson

# MarkovNLPJson to use postag during modeling
mkov = MarkovNLPJson(order=3)
model = "questions"
path = f"{model}_w{mkov.order}.mkovjson"

with open(f"sample_datasets/{model}.txt") as f:
    for line in f.readlines():
        if not line.strip():
            continue
        label, question = line.split(" ", 1)
        normalized = question.replace("?", "").lower().strip()
        mkov.add_string(normalized)

for i in range(10):
    print(mkov.generate_string())
    # how is the election of a new pope announced to the world
    # where can i find out what is allowed to claim as a contibution for income tax purposes
    # what is an arab strap
    # how many stradivarius violins were ever made
    # how many beatles records went # 1
    # hazmat stands for what
    # how does the tail affect the flight of a kite
    # what 's the most powerful microscope and how big is a quart
    # how many colors was the 1940s collectible called a donald duck rubber boat
    # what is the address for the main government office in rome italy


for i in range(10):
    print(mkov.generate_tagged_string()) # since we used postag, we also get tagged output
    # [('what', 'WP'), ('is', 'VBZ'), ('grenada', 'NN'), ("'s", 'POS'), ('main', 'JJ'), ('commodity', 'NN'), ('export', 'NN')]
    # [('who', 'WP'), ('are', 'VBP'), ('cartoondom', 'NN'), ("'s", 'POS'), ('super', 'JJ'), ('six', 'CD')]
    # [('what', 'WP'), ('is', 'VBZ'), ('the', 'DT'), ('size', 'NN'), ('of', 'IN'), ('argentina', 'NN')]
    # [('who', 'WP'), ('is', 'VBZ'), ('the', 'DT'), ('incredible', 'JJ'), ('hulk', 'NN')]
    # [('what', 'WP'), ('1920s', 'CD'), ('cowboy', 'NN'), ('star', 'NN'), ('rode', 'NN'), ('tony', 'IN'), ('the', 'DT'), ('wonder', 'NN'), ('horse', 'NN')]
    # [('when', 'WRB'), ('was', 'VBD'), ('the', 'DT'), ('battle', 'NN'), ('of', 'IN'), ('the', 'DT'), ('somme', 'NN'), ('fought', 'NN')]
    # [('what', 'WP'), ('judith', 'NN'), ('rossner', 'NN'), ('novel', 'NN'), ('was', 'VBD'), ('made', 'VBN'), ('into', 'IN'), ('a', 'DT'), ('film', 'NN')]
    # [('colin', 'NN'), ('powell', 'NN'), ('is', 'VBZ'), ('most', 'RBS'), ('famous', 'JJ'), ('for', 'IN')]
    # [('how', 'WRB'), ('do', 'VB'), ('you', 'PRP'), ('exterminate', 'VB'), ('bees', 'NNS'), ('that', 'WDT'), ('are', 'VBP'), ('in', 'IN'), ('the', 'DT'), ('un', 'NN')]
    # [('what', 'WP'), ('u.s.', 'JJ'), ('vice-president', 'JJ'), ('once', 'RB'), ('declared', 'VBN'), (':', ':'), ('if', 'IN'), ('you', 'PRP'), ("'ve", 'VBP'), ('seen', 'VBN'), ('them', 'PRP'), ('all', 'DT')]