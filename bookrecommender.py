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

threebodyproblem = parse_paper("/Users/samkaufman/PycharmProjects/ExamGPT/Books/The Three-Body Problem by Liu Cixin (z-lib ( PDFDrive ).pdf")
hammingbook = parse_paper("/Users/samkaufman/PycharmProjects/ExamGPT/Books/Hamming-TheArtOfDoingScienceAndEngineering.pdf")
#campbell = parse_paper("/Users/samkaufman/PycharmProjects/ExamGPT/Books/(Campbell Biology Series) Lisa A. Urry, Micheal L. Cain, Steven A. Wasserman, Peter V. Minorsky, Rebecca B. Orr, Neil A. Campbell - Campbell Biology, 12th Edition-Pearson (2020).pdf")
fallapart = parse_paper("/Users/samkaufman/PycharmProjects/ExamGPT/Books/things fall apart ( PDFDrive ).pdf")
zerotoone = parse_paper("/Users/samkaufman/PycharmProjects/ExamGPT/Books/021.pdf")
windup = parse_paper("/Users/samkaufman/PycharmProjects/ExamGPT/Books/The Wind-Up Bird Chronicle ( PDFDrive ).pdf")
norwood = parse_paper("/Users/samkaufman/PycharmProjects/ExamGPT/Books/Haruki Murakami - Norwegian Wood.pdf")

def comparebooks(df1, df2, n=3, pprint=True):
    #totalsimilarity = []
    #for i in range(len(df2.index)):
    #    thissimilarity = df1.embeddings.apply(lambda x: cosine_similarity(x, df2["embeddings"].iloc(i)))
    #    totalsimilarity.append(thissimilarity)

    totalsimilarity= [cosine_similarity(p0,p1) for p0,p1 in zip(df1["embeddings"], df2["embeddings"])]
    similarity = sum(totalsimilarity)/len(totalsimilarity)
    return similarity


# norwooddf = pd.DataFrame(norwood[0])
# norwooddf = norwooddf[norwooddf['text'] != '']
# embedding_model = "text-embedding-ada-002"
# embeddings = norwooddf.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
# norwooddf["embeddings"] = embeddings
#
# windupdf = pd.DataFrame(windup[0])
# windupdf = windupdf[windupdf['text'] != '']
# embedding_model = "text-embedding-ada-002"
# embeddings = windupdf.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
# windupdf["embeddings"] = embeddings
#
# zerotoonedf = pd.DataFrame(zerotoone[0])
# zerotoonedf = zerotoonedf[zerotoonedf['text'] != '']
# embedding_model = "text-embedding-ada-002"
# embeddings = zerotoonedf.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
# zerotoonedf["embeddings"] = embeddings
#
# fallapartdf = pd.DataFrame(fallapart[0])
# fallapartdf = fallapartdf[fallapartdf['text'] != '']
# embedding_model = "text-embedding-ada-002"
# embeddings = fallapartdf.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
# fallapartdf["embeddings"] = embeddings
#
# hammingbookdf = pd.DataFrame(hammingbook[0])
# hammingbookdf = hammingbookdf[hammingbookdf['text'] != '']
# embedding_model = "text-embedding-ada-002"
# embeddings = hammingbookdf.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
# hammingbookdf["embeddings"] = embeddings
#
# threebodyproblemdf = pd.DataFrame(threebodyproblem[0])
# threebodyproblemdf = threebodyproblemdf[threebodyproblemdf['text'] != '']
# embedding_model = "text-embedding-ada-002"
# embeddings = threebodyproblemdf.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
# threebodyproblemdf["embeddings"] = embeddings
with open('objs.pkl', 'rb') as file:
    # Call load method to deserialze
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
    # Call load method to deserialze
    bookembeddings2 = pickle.load(file, encoding="bytes")
[crimeandpundf, leanstartupdf, f451df, dunedf, hamletdf, orwelldf, romeodf, wanderingdf, merchantdf] = bookembeddings2

booklist = [threebodyproblemdf, hammingbookdf, norwooddf, fallapartdf, zerotoonedf, windupdf, crimeandpundf, leanstartupdf, f451df, dunedf, hamletdf, orwelldf, romeodf, wanderingdf, merchantdf]

for i in range(len(booklist)):
    for j in range(len(booklist)):
        print((booklist[i]["name"]) + " " + booklist[j]["name"] + " " + str(comparebooks(booklist[i], booklist[j])))

def say(msg = "Get over here", voice = "Victoria"):
    os.system(f'say -v {voice} {msg}')

say()