from snorkel.labeling import labeling_function
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
import pandas as pd
from sentence_transformers import SentenceTransformer, util


#df_defi = pd.read_csv('data/NCF_Definitions_preprocessed.csv')
df_defi = pd.read_csv('data/climate_definitions.csv')
df = pd.read_csv('data/metadata0.csv')

#df = pd.read_csv('data/data_NCF_all.csv')
#df = pd.read_csv('../data/data_NCF/metadata0.csv')
#df = pd.read_csv('../data_fulltext/data_climate/text3.csv')
#df = df.iloc[0:5000,:]
#df = df.head(300)

df_labeled = pd.DataFrame()
df_labeled['abstract'] = df['abstract']
#df_labeled['text'] =  df['text']
communications = 1
ABSTAIN = 0


i = 0
@labeling_function()
def lf_def_1(x):
    i = 0
    embedder = SentenceTransformer('multi-qa-distilbert-cos-v1')
    list_key = df_defi['Definition'].iloc[i]
    def_embedding = embedder.encode(list_key, convert_to_tensor=True)
    corpus_embeddings = embedder.encode(x, convert_to_tensor=True)  
    score = util.pytorch_cos_sim(def_embedding, corpus_embeddings)[0]
    if score > 0.5:
      return communications  
    return ABSTAIN  

@labeling_function()
def lf_def_2(x):
    i = 0
    embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    list_key = df_defi['Definition'].iloc[i]
    def_embedding = embedder.encode(list_key, convert_to_tensor=True)
    corpus_embeddings = embedder.encode(x, convert_to_tensor=True)  
    score = util.pytorch_cos_sim(def_embedding, corpus_embeddings)[0]
    if score > 0.5:
      return communications  
    return ABSTAIN  

@labeling_function()
def lf_def_3(x):
    i = 0
    embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    list_key = df_defi['Definition'].iloc[i]
    def_embedding = embedder.encode(list_key, convert_to_tensor=True)
    corpus_embeddings = embedder.encode(x, convert_to_tensor=True)  
    score = util.pytorch_cos_sim(def_embedding, corpus_embeddings)[0]
    if score > 0.5:
      return communications  
    return ABSTAIN  

@labeling_function()
def lf_def_4(x):
    i = 0
    embedder = SentenceTransformer('all-mpnet-base-v2')
    list_key = df_defi['Definition'].iloc[i]
    def_embedding = embedder.encode(list_key, convert_to_tensor=True)
    corpus_embeddings = embedder.encode(x, convert_to_tensor=True)  
    score = util.pytorch_cos_sim(def_embedding, corpus_embeddings)[0]
    if score > 0.5:
      return communications  
    return ABSTAIN  

@labeling_function()
def lf_def_5(x):
    i = 0
    embedder = SentenceTransformer('all-distilroberta-v1')
    list_key = df_defi['Definition'].iloc[i]
    def_embedding = embedder.encode(list_key, convert_to_tensor=True)
    corpus_embeddings = embedder.encode(x, convert_to_tensor=True)  
    score = util.pytorch_cos_sim(def_embedding, corpus_embeddings)[0]
    if score > 0.5:
      return communications  
    return ABSTAIN  

@labeling_function()
def lf_def_6(x):
    i = 0
    embedder = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
    list_key = df_defi['Definition'].iloc[i]
    def_embedding = embedder.encode(list_key, convert_to_tensor=True)
    corpus_embeddings = embedder.encode(x, convert_to_tensor=True)  
    score = util.pytorch_cos_sim(def_embedding, corpus_embeddings)[0]
    if score > 0.5:
      return communications  
    return ABSTAIN  

@labeling_function()
def lf_def_7(x):
    i = 0
    embedder = SentenceTransformer('msmarco-distilbert-base-v4')
    list_key = df_defi['Definition'].iloc[i]
    def_embedding = embedder.encode(list_key, convert_to_tensor=True)
    corpus_embeddings = embedder.encode(x, convert_to_tensor=True)  
    score = util.pytorch_cos_sim(def_embedding, corpus_embeddings)[0]
    if score > 0.5:
      return communications  
    return ABSTAIN  


# Define the set of labeling functions (LFs)
lfs = [lf_def_1, lf_def_2, lf_def_3, lf_def_4, lf_def_5, lf_def_6, lf_def_7] #, lf_textblob_polarity]



# Apply the LFs to the unlabeled training data
applier = PandasLFApplier(lfs)
L_train = applier.apply(df_labeled)

# Train the label model and compute the training labels
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)

df_labeled['paper_id'] = df['paper_id']
df_labeled['abstract'] = df['abstract']
df_labeled[str(i)] = label_model.predict(L=L_train, tie_break_policy="abstain")
#df_labeled.to_csv('../data_fulltext/data_climate_label/text_3_label'+ str(i)+'.csv', index=False)
df_labeled.to_csv('results_Climate_0.5/metadata0_def_'+ str(i)+'.csv', index=False)

