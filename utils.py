import os
from openai import OpenAI, AzureOpenAI
from google import genai
from google.genai import types
import subprocess
from itertools import islice


SYSTEM_PROMPT = """"""

def get_lines(file_path, start, end):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in islice(f, start - 1, end)]


def count_lines(file_path):
    return int(subprocess.check_output(["wc", "-l", file_path]).split()[0])


def openai_generate(model_name, content):
    client = OpenAI()

    response = []

    for part in content:
        partial_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": part
                        }
                    ]
                }
            ],
            response_format={
                "type": "text"
            },
        )

        response.append(partial_response.choices[0].message.content)

    return response


def azure_generate(model_name, content):
    client = AzureOpenAI(  
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version=os.getenv("OPENAI_API_VERSION"),
    )

    response = []
    for part in content: 
        partial_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": part
                        }
                    ]
                }
            ],
            stream=False
        )
        response.append(partial_response.choices[0].message.content)

    return response

def gemini_generate(model_name, content):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    response = []
    for part in content:
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"""{part}"""),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text=f"""{SYSTEM_PROMPT}"""),
            ],
        )

        partial_response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        )

        response.append(partial_response.text.strip())

    return response
