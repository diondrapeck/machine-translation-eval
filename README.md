# LLM Translation Evaluation

This repository contains a benchmarking script for evaluating different machine translation models using BLEU score. The script uses the [sacrebleu](https://github.com/mjpost/sacrebleu) library to download test datasets, generate translations using your choice of models, and calculates BLEU scores to compare the performance of the models on the translation task.

All you need are model deployments in Gemini, Azure OpenAI, and/or OpenAI APIs.

## How to Use

1. Clone the repository:

    ```
    git clone https://github.com/diondrapeck/machine-translation-eval.git
    cd machine-translation-eval
    ```

2. Create and activate a virtual environment:

    ```
    python3 -m venv venv
    source venv/bin/activate  # unix
    venv\Scripts\Activate.ps1 # Powershell
    ```

3. Install the required packages:

    ```
    pip install -r requirements.txt
    ```

4. Set up your environment variables by creating a .env file in the root directory of the project and adding your API keys. You only need to supply variables for the service providers your models are deployed on:

    ```env
    OPENAI_API_KEY = <>
    AZURE_OPENAI_ENDPOINT = <>
    AZURE_OPENAI_API_KEY = <>
    OPENAI_API_VERSION = <>
    GEMINI_API_KEY = <>
    ```

5. Update the candidate_models.json with your model deployments you want to test the translation on.

6. Run the benchmarking script with the desired arguments:

```
python model-benchmark.py --source_language en --target_language fr --chunk_size 100
```

    Arguments
    --target-language: 2 character ISO code of the target language to translate into.
    --source-language: 2 character ISO code of the source language to translate from. Defaults to "en", for English
    --chunk_size: The size of the text sample from the dataset to use for translation task. Defaults to 500 lines.