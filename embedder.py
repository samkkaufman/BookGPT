import pandas
from openai.embeddings_utils import get_embedding, cosine_similarity
import streamlit as st
import pandas as pd
import openai
import os
import matplotlib
import plotly
import scipy
import math
import sklearn
import pickle
from pypdf import PdfReader

def parse_paper(path):
    print("Parsing paper")
    reader = PdfReader(path)
    number_of_pages = len(reader.pages)
    print(f"Total number of pages: {number_of_pages}")
    paper_text = []
    for i in range(number_of_pages):
        page = reader.pages[i]
        page_text = page.extract_text()
        paper_text.append(
            {'text':page_text}
        )
    paper_text_string = ''
    for i in paper_text:
        paper_text_string = paper_text_string + ' ' + i['text']
    result = [paper_text, paper_text_string]
    return result

openai.api_key = "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"

crimeandpun = parse_paper("path")
leanstartup = parse_paper("path")
f451 = parse_paper("path")
dune = parse_paper("path")
hamlet = parse_paper("path")
orwell = parse_paper("path")
romeo = parse_paper("path")
wandering = parse_paper("path")
merchant = parse_paper("path")

crimeandpundf = pd.DataFrame(crimeandpun[0])
crimeandpundf = crimeandpundf[crimeandpundf['text'] != '']
embedding_model = "text-embedding-ada-002"
embeddings = crimeandpundf.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
crimeandpundf["embeddings"] = embeddings
crimeandpundf["name"] = "crimeandpun"

leanstartupdf = pd.DataFrame(leanstartup[0])
leanstartupdf = leanstartupdf[leanstartupdf['text'] != '']
embedding_model = "text-embedding-ada-002"
embeddings = leanstartupdf.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
leanstartupdf["embeddings"] = embeddings
leanstartupdf["name"] = "leanstartup"

f451df = pd.DataFrame(f451[0])
f451df = f451df[f451df['text'] != '']
embedding_model = "text-embedding-ada-002"
embeddings = f451df.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
f451df["embeddings"] = embeddings
f451df["name"] = "f451"

dunedf = pd.DataFrame(dune[0])
dunedf = dunedf[dunedf['text'] != '']
embedding_model = "text-embedding-ada-002"
embeddings = dunedf.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
dunedf["embeddings"] = embeddings
dunedf["name"] = "dune"

hamletdf = pd.DataFrame(hamlet[0])
hamletdf = hamletdf[hamletdf['text'] != '']
embedding_model = "text-embedding-ada-002"
embeddings = hamletdf.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
hamletdf["embeddings"] = embeddings
hamletdf["name"] = "hamlet"

orwelldf = pd.DataFrame(orwell[0])
orwelldf = orwelldf[orwelldf['text'] != '']
embedding_model = "text-embedding-ada-002"
embeddings = orwelldf.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
orwelldf["embeddings"] = embeddings
orwelldf["name"] = "orwell"

romeodf = pd.DataFrame(romeo[0])
romeodf = romeodf[romeodf['text'] != '']
embedding_model = "text-embedding-ada-002"
embeddings = romeodf.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
romeodf["embeddings"] = embeddings
romeodf["name"] = "romeo"

wanderingdf = pd.DataFrame(wandering[0])
wanderingdf = wanderingdf[wanderingdf['text'] != '']
embedding_model = "text-embedding-ada-002"
embeddings = wanderingdf.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
wanderingdf["embeddings"] = embeddings
wanderingdf["name"] = "wandering"

merchantdf = pd.DataFrame(merchant[0])
merchantdf = merchantdf[merchantdf['text'] != '']
embedding_model = "text-embedding-ada-002"
embeddings = merchantdf.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
merchantdf["embeddings"] = embeddings
merchantdf["name"] = "merchant"


with open('objs2.pkl', 'wb') as file:
    myvar = [crimeandpundf, leanstartupdf, f451df, dunedf, hamletdf, orwelldf, romeodf, wanderingdf, merchantdf]
    pickle.dump(myvar, file)

def say(msg = "Get over here", voice = "Victoria"):
    os.system(f'say -v {voice} {msg}')

say()
