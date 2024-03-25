# bertopic-base-finetuned-ecommerce-ergonomics-chair

<!-- Provide a quick summary of what the model is/does. [Optional] -->
This a Bertopic base model , with embedding_model &#34;sentence-transformers/all-mpnet-base-v2&#34;, finetuned  by embedding on ergonomics chair product reviews in English language. 
This model is intended for direct use as a topic/cluster model for product reviews in ergonomics chair minor field, or for further finetuning on related clustering analysis tasks.
I replaced its head with our customer reviews to fine-tune it on 50,000+ rows of training set while validating it on 35,000 rows of dev set. 
- **Developed by:** ilaria Huang
- **Model type:** Language model
- **Language(s) (NLP):** en
- **License:** mit
- **Parent Model:** sentence-transformers/all-mpnet-base-v2; BERTopic; phrasemachine
- **Resources for more information:** 
    - [GitHub Repo](https://github.com/yufei-ilariahuang/Ergonomics-Chair-project.git)
    - [Hugging Face]model: https://huggingface.co/liaHa/bertopic-base-finetuned-ecommerce-ergonomics-chair
# How to Get Started with the Model
Use the code below to get started with the model.
```python
loaded_model = BERTopic.load("bertopic-base-finetuned-ecommerce-ergonomics-chair")
```
# Training Procedure
#### Preprocessing
sentence_model encoded with remove duplicated phrases of ergonomics chair customer review data, preprocessed by phrasemachine (https://github.com/slanglab/phrasemachine)
#### Training Data
<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->
I use the customer review data from multiple e-commerce websites to fine-tune our model. 
You download all the raw datasets from: [dataset](https://huggingface.co/datasets/liaHa/Ergonomics_Chiar_Customer_Viewdata_E-commerse)
#### Pre-training
I use the pretrained sentence-transformers/all-mpnet-base-v2 model and BERTopic model. 
Please refer to the model card(https://huggingface.co/sentence-transformers/all-mpnet-base-v2),(https://huggingface.co/blog/bertopic) for more detailed information about the pre-training procedure.
#### Hyperparameters
The following hyperparameters were used during training:
<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->
- representation_model = MaximalMarginalRelevance(diversity=0.3) # for diversity in topic name
- vectorizer_model = CountVectorizer
- sentence_model = SentenceTransformer("all-mpnet-base-v2")
- hdbscan_model = HDBSCAN(min_cluster_size=50,
                        metric='euclidean',
                        cluster_selection_method='eom',
                        prediction_data=True)
- umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
# Framework versions
- BERTopic version: 0.16.0
- Python version: 3.11.1
