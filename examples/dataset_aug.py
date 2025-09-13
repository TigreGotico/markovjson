import os
from datasets import load_dataset
from markovjson import MarkovWordJson

# Define the dataset and model file paths
DATASET_NAME = "Jarbas/music_queries_metal_bands"
MODEL_PATH = "metal_queries.mkovjson"
MODEL_ORDER = 3

# Initialize the Markov chain model
mkov = MarkovWordJson(order=MODEL_ORDER)

if os.path.isfile(MODEL_PATH):
    print(f"Loading existing model from {MODEL_PATH}...")
    mkov.load(MODEL_PATH)
else:
    print(f"Model not found. Training on '{DATASET_NAME}'...")
    # Load the dataset from Hugging Face
    dataset = load_dataset(DATASET_NAME, split="train")

    # Train the model on the 'synthetic_query' column
    for data in dataset:
        query = data.get("synthetic_query")
        if query:
            mkov.add_string(query)

    # Save the trained model for future use
    mkov.save(MODEL_PATH)
    print(f"Training complete. Model saved to {MODEL_PATH}.")

print("\n--- Generating new queries ---")
# Generate new queries based on the trained model
for i in range(50):
    new_query = mkov.generate_string()
    print(f"Generated Query {i+1}: {new_query}")
