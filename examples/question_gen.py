from markovjson import MarkovNLPJson

mkov = MarkovNLPJson(order=3)
model = "questions"
path = f"{model}_w{mkov.order}.mkovjson"

with open(f"datasets/{model}.txt") as f:
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
