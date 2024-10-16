import pandas as pd
import numpy as np

import string
import re
from collections import Counter
from tqdm import tqdm

from sklearn.utils import shuffle
import statistics as st

#######################################
## Data Merging, Preprocessing, and Cleanup

# Load and merge source data
label_desc=pd.read_excel('Classification_Groups.xlsx')
label_desc['Label'] = label_desc['Label'].apply(lambda x: x.strip() if isinstance(x, str) else x)

inputs_ex=pd.read_csv('examples_cleaned.csv')
inputs_ex['Label'] = inputs_ex['Label'].apply(lambda x: x.strip() if isinstance(x, str) else x)

merged_df=inputs_ex.merge(label_desc, how='left',on='Label').copy()


# Clean and add filters for context
Base_df=merged_df.dropna(subset=['Category']).copy()
Base_df.dropna(subset=['Text'],inplace=True)
Base_df['P_text']=0
Base_df['C_text']=0
Base_df['D_text']=0
P='Parent:'
C='Child:'
D='Dyad:'
Base_df['P_text']=np.where(Base_df.Text.str.find(P)==-1,0,1)
Base_df['C_text']=np.where(Base_df.Text.str.find(C)==-1,0,1)
Base_df['D_text']=np.where(Base_df.Text.str.find(D)==-1,0,1)


# Isolate Parent Only Text
Only_Parent=Base_df[(Base_df['P_text']==1) & (Base_df['C_text']==0) & (Base_df['D_text']==0)].copy()
Only_Parent['Text_clean']=Only_Parent['Text'].str.replace(P,"")

# Isolate Parent With Child (no dyad)
ChildwParent=Base_df[(Base_df['P_text']==1) & (Base_df['C_text']==1)].copy()
ChildwParent['Text_split']=ChildwParent['Text'].str.split(P)

# Some processinga nd cleanup tasks
for index, row in ChildwParent.iterrows():
    ChildwParent.loc[index,'Text_clean']=row['Text_split'][1]

ChildwParent['C_text2']=np.where(ChildwParent.Text_clean.str.find(C)==-1,0,1)
ChildwParent2=ChildwParent[ChildwParent['C_text2'] != 1].copy()
ChildwParent2.drop(columns=['C_text2','Text_split'],inplace=True)


# Concatenate into Parent only, Parent + Child DF
PC_Merged=pd.concat([Only_Parent,ChildwParent2])

#Shuffle (if needed)
PC_Merged = shuffle(PC_Merged)
PC_Merged.reset_index(drop=True, inplace=True)
PC_Merged.head()


#######################################
## Data Tokenization


# Some cleanup vocab and text processing functions

def build_vocab(doc):
    tokens=doc.split()
    re_punc=re.compile('[%s]' % re.escape(string.punctuation))
    tokens=[re_punc.sub('',w) for w in tokens]
    tokens=[word for word in tokens if word.isalpha()]
    tokens=[word.lower() for word in tokens]
    #stop_words=[]
    return tokens


def clean_text(doc, vocab):
    tokens=doc.split()
    re_punc=re.compile('[%s]' % re.escape(string.punctuation))
    tokens=[re_punc.sub('',w) for w in tokens]
    tokens=[word for word in tokens if word.isalpha()]
    #tokens=[w for w in tokens if w in vocab]
    tokens=[word.lower() for word in tokens]
    tokens=' '.join(tokens)
    #stop_words=[]
    return tokens



# Clean text, convert to tokens for further processing

vocab=Counter()
text_inputs=list(PC_Merged['Text_clean'])

for t in text_inputs:
    vtokens=build_vocab(t)
    vocab.update(vtokens)

#print(len(vocab))
#print(vocab.most_common(30))

interactions=list()
for row in PC_Merged.itertuples():
    doc=row.Text_clean
    tokens=clean_text(doc, vocab)
    interactions.append(tokens)
#print(interactions[:5])

words = ' '.join(interactions)
words = words.split()
counter = Counter(words)
vocab = sorted(counter, key=counter.get, reverse=True)
int2word = dict(enumerate(vocab, 1))
int2word[0] = '<PAD>'
word2int = {word: id for id, word in int2word.items()}
text_enc = [[word2int[word] for word in interaction.split()] for interaction in tqdm(interactions)]


# Check stats on excerpt length
ls=set()
for s in text_enc:
    ls.add(len(s))
print("max Length Sentence: ", max(ls))
print("Mean Length Sentence: ",st.mean(ls))
print("Median Length Sentence: ",st.median(ls))
print("STD  Sentence: ",st.stdev(ls))



# Padding

def pad_features(textin, pad_id, seq_length=128):
    features = np.full((len(textin), seq_length), pad_id, dtype=int)
    for i, row in enumerate(textin):
        # if seq_length < len(row) then text will be trimmed
        features[i, :len(row)] = np.array(row)[:seq_length]
    return features

#seq_length = 256
seq_length=15

features = pad_features(text_enc, pad_id=word2int['<PAD>'], seq_length=seq_length)
assert len(features) == len(text_enc)
assert len(features[0]) == seq_length

print(features[:5, :])