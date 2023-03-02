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





with open('objs.pkl', 'rb') as file:

    bookembeddings = pickle.load(file, encoding="bytes")

[campbelldf, norwooddf, windupdf, zerotoonedf, fallapartdf, hammingbookdf, threebodyproblemdf] = bookembeddings

campbelldf["name"] = "campbell"
norwooddf["name"] = "norwood"
windupdf["name"] = "windup"
zerotoonedf["name"] = 'zerotoone'
fallapartdf["name"] = "fallapart"
hammingbookdf["name"] = "hammingbook"
threebodyproblemdf["name"] = "3bp"

with open('objs2.pkl', 'rb') as file:

    bookembeddings2 = pickle.load(file, encoding="bytes")
[crimeandpundf, leanstartupdf, f451df, dunedf, hamletdf, orwelldf, romeodf, wanderingdf, merchantdf] = bookembeddings2

booklist = [threebodyproblemdf, hammingbookdf, norwooddf, fallapartdf, zerotoonedf, windupdf, crimeandpundf, leanstartupdf, f451df, dunedf, hamletdf, orwelldf, romeodf, wanderingdf, merchantdf]

for i in range(len(booklist)):
    for j in range(len(booklist)):
        print((booklist[i]["name"]) + " " + booklist[j]["name"] + " " + str(comparebooks(booklist[i], booklist[j])))

