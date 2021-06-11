
from transformers import pipeline
classifier = pipeline('sentiment-analysis')


def sentiment_analysis(input_text):
    result_sent = classifier(input_text)
    return result_sent


from transformers import BertTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np

tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

goemotions = MultiLabelPipeline(
    model=model,
    tokenizer=tokenizer,
    threshold=0.3
)


def Sentiments_analysis(essay_input):
    
    re_text = essay_input.split(".")

    
    def cleaning(datas):

        fin_datas = []

        for data in datas:
        
            only_english = re.sub('[^a-zA-Z]', ' ', data)
        
            fin_datas.append(only_english)

        return fin_datas

    texts = cleaning(re_text)

    emo_re = goemotions(texts)

    emo_all = []
    for list_val in range(0, len(emo_re)):
        emo_all.append((emo_re[list_val]['labels']))
        

    from pandas.core.common import flatten 
    flat_list = list(flatten(emo_all))

    unique_re = set(flat_list) 


    return unique_re




input_text = """i am happy today."""
print(Sentiments_analysis(input_text))