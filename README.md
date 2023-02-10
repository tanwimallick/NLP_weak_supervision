# A weakly supervised NLP approach to analyze scientific literature on climate change ad critical infrastructure 

We designed a method for programmatic labeling of a large corpus using weak supervision, allowing the dataset to be labeled with minimal human involvement. Programmatic labeling with weak supervision is accomplished by defining labeling functions that  capture semantic similarity instead of patterns, heuristics, or rules for labeling. The labeling functions in our weak supervision method evaluate the semantic similarity between the definitions and unlabeled documents. To increase performance by making the method more robust, we construct the labeling functions for the first time utilizing multiple semantic embedding techniques. The semantic embedding models have different architectures and  are pretrained on different general-purpose datasets. Therefore, a single embedding technique is not sufficient to label our climate and NCF dataset. Hence, we define multiple labeling functions using different embedding techniques to cover different aspects of the data. The weak supervision model generates the probabilistic labels by maximizing the overlap and minimizing the conflict among the multiple labeling functions using probabilistic modeling. 

## Requirements
- snorkel >= 0.9.8
- sentence_transformers >= 2.2.0
- numpy >= 1.20.3
- pandas >= 1.2.4

## Data 

The cliamte and critical infrastructure data is available under data folder in the file data_NCF_climate_all.csv

Simply run the following command to generate the label for the first climate definition.

python snorkel_label0.py
