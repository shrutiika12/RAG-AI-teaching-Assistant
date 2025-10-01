import requests
import os
import json
import pandas as pd

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "bge-m3",
        "prompt": text_list
    })
# The keyword "embedding" is defined by Ollama’s API, not by Python.
    embedding = r.json()['embedding'] 
    return embedding

# embedding (a number representation of words).
a=create_embedding("cat sat on the mat")
print(a)

