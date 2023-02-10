import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import time

start_time = time.time()

embedder = SentenceTransformer('all-distilroberta-v1')
#df_def = pd.read_csv('climate_definitions.csv')
df_def = pd.read_csv('NCF_Definitions_preprocessed.csv')
#df_def = df_def.dropna()
#df_def = df_def.reset_index(drop=True)

ind =0
#df_doc = pd.read_csv('metadata/abstract/metadata' + str(ind) +'.csv')
df_doc = pd.read_csv('data_climate/metadata' + str(ind) +'.csv')
df_doc = df_doc.dropna()
df_doc = df_doc.reset_index(drop=True)

def_embedding = []
for i in range(55): #df_def.shape[0]):
    def_embedding.append(embedder.encode(df_def['Definition_org'][i], convert_to_tensor=True))
    #def_embedding.append(embedder.encode(df_def['detailed'][i], convert_to_tensor=True))

text_embedding = []
for i in range(df_doc.shape[0]): #41550
#    print(i)
    text_embedding.append(embedder.encode(df_doc['abstract'][i], convert_to_tensor=True))


max_sim = []
for i in range(len(text_embedding)):
    scores = []
    for j in range(len(def_embedding)):
        score = util.pytorch_cos_sim(def_embedding[j], text_embedding[i])[0]
        score = float(score)
        scores.append(score)
        print(score)
    max_sim.append(np.array(scores).max())

df = pd.DataFrame()
df['abstract'] = df_doc['abstract'][0:len(max_sim)]
df['score'] = max_sim
df['year'] = df_doc['year'][0:len(max_sim)]
df['field'] = df_doc['field'][0:len(max_sim)]
df['title'] = df_doc['title'][0:len(max_sim)]
df['paper_id'] = df_doc['paper_id'][0:len(max_sim)]
df['authors'] = df_doc['authors'][0:len(max_sim)]

df = df.loc[df['score'] > 0.4]
df.to_csv('NCF_metadata' + str(ind) +'.csv')

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")
#df.to_csv('data_NCF_climate/metadata' + str(ind) +'.csv')
