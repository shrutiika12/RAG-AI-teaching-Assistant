import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
import joblib
import json

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream":False
    })
    response=r.json()
    print(response)
    return response

df=joblib.load('embeddings.joblib')
incoming_query=input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

#find similarities of que embeddings with other embeddings
similarities=cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
print(similarities)
max_ind=similarities.argsort()[::-1][0:3]
print(max_ind)
new_df=df.loc[max_ind]
print(new_df[["title","number","text"]])

prompt=f'''I am teaching web development using web development course. Here are video subtitle chunks containing video title,video number, start time in second, end time in seconds, the text at that time :

{new_df[["title","number","start","end","text"]].to_json(orient="records")}
-------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks , you have to answer where and how much content is taught in which video (in which and at what timwstamp) and guide the user to go to that particular video. If user asks unrelated quetsion ,tell him that you can only answer questions related to the course'''

with open ("prompt.txt" ,"w") as f:
    f.write(prompt)

response=inference(prompt) ["response"]
print(response)

with open ("response.txt" ,"w") as f:
    f.write(response)
