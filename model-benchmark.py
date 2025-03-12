import os
import random
import json
import argparse
from dotenv import load_dotenv

from sacrebleu.utils import get_available_testsets_for_langpair, download_test_set
from sacrebleu import corpus_bleu

from utils import gemini_generate, openai_generate, azure_generate,count_lines, get_lines


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--source_language", type=str, default="en", help="Source language ISO code")
parser.add_argument("--target_language", type=str, help="Target language ISO code")
parser.add_argument("--chunk_size", type=int, default=500, help="Size of text chunks for translation")
args = parser.parse_args()

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Set languages
SOURCE_LANGUAGE = args.source_language
TARGET_LANGUAGE = args.target_language
LANG_PAIR = f"{SOURCE_LANGUAGE}-{TARGET_LANGUAGE}"
SYSTEM_PROMPT = f"""You are an expert text translator translating text from ISO code {SOURCE_LANGUAGE} to ISO code {TARGET_LANGUAGE}. Do not return any intermediate steps, only return the translation."""
TEXT_SAMPLE_SIZE =args.chunk_size

# Grab models
with open("candidate_models.json", "r") as file:
	candidate_models = json.load(file)

# Get reference dataset
available_testsets = get_available_testsets_for_langpair(LANG_PAIR)
test_set_name = random.choice(available_testsets)
print(f"Chose test set {test_set_name} for {LANG_PAIR} translation")

test_set = download_test_set(test_set=test_set_name, langpair=LANG_PAIR)
test_original, test_translation = test_set[0], test_set[1]
chunk_start = random.randint(1, count_lines(test_original) - TEXT_SAMPLE_SIZE)
original_text = get_lines(file_path=test_original, start=chunk_start, end=(chunk_start + TEXT_SAMPLE_SIZE))
reference_translation = get_lines(file_path=test_translation, start=chunk_start, end=(chunk_start + TEXT_SAMPLE_SIZE))
print(f"Selecting lines {chunk_start} to {chunk_start + TEXT_SAMPLE_SIZE} of {test_original} for translation")

# Generate translations
results = {}
for model in candidate_models["gemini"]:
	results[model] = gemini_generate(model_name=model, content=original_text, system_prompt=SYSTEM_PROMPT)

for model in candidate_models["openai"]:
	results[model] = openai_generate(model_name=model, content=original_text, system_prompt=SYSTEM_PROMPT)

for model in candidate_models["azure"]:
	results[model] = azure_generate(model_name=model, content=original_text, system_prompt=SYSTEM_PROMPT)

# Calculate BLEU scores
for model, translation in results.items():
	bleu_score = corpus_bleu(translation, [reference_translation])
	print(f"{model} BLEU Score: {bleu_score.score:.2f}")