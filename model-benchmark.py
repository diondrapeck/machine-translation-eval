import os
import random
import json
from dotenv import load_dotenv

from sacrebleu.utils import get_available_testsets_for_langpair, download_test_set
from sacrebleu import corpus_bleu

from utils import gemini_generate, openai_generate, azure_generate,count_lines, get_lines


# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Set languages
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "es"
LANG_PAIR = f"{SOURCE_LANGUAGE}-{TARGET_LANGUAGE}"

# Set model variables
CHUNK_SIZE = 100

# Grab models
with open("candidate_models.json", "r") as file:
	candidate_models = json.load(file)

# Get reference dataset
available_testsets = get_available_testsets_for_langpair(LANG_PAIR)
test_set_name = random.choice(available_testsets)
print(f"Chose test set {test_set_name}")

test_set = download_test_set(test_set=test_set_name, langpair=LANG_PAIR)
test_original, test_translation = test_set[0], test_set[1]
chunk_start = random.randint(1, count_lines(test_original) - CHUNK_SIZE)
test_text = get_lines(file_path=test_original, start=chunk_start, end=(chunk_start + CHUNK_SIZE))
reference_text = get_lines(file_path=test_translation, start=chunk_start, end=(chunk_start + CHUNK_SIZE))
print(f"Selecting lines {chunk_start} to {chunk_start + CHUNK_SIZE} of {test_original} for translation")

# Generate translations
results = {}
for model in candidate_models["gemini"]:
	results[model] = gemini_generate(model_name=model, content=test_text)

for model in candidate_models["openai"]:
	results[model] = openai_generate(model_name=model, content=test_text)

for model in candidate_models["azure"]:
	results[model] = azure_generate(model_name=model, content=test_text)



# Calculate BLEU scores
for model, translation in results.items():
	bleu_score = corpus_bleu(translation, [reference_text])
	print(f"{model} BLEU Score: {bleu_score.score:.2f}")