# Extraction pipelines vaccine hesitancy
### Repository for the Lagrange Scholarships Projects about Vaccine Hesitancy - Extraction pipelines  

In this project, we developed two high precision rule-based extraction pipelines able to classify text with respect to vaccination behaviors and experiences. The items we tracked are (i) adherence to the recommended or alternative vaccination schedule and (ii) mentions of positive or negative experiences with adverse events following immunization (AEFI).  

The two pipelines share the same workflow and work at the level of sentences. They are made up by a filter and a classifier. The filter identifies sentences which contain information relevant to the item under consideration by using a combination of rules based on the occurrence of certain keyword with specific syntactic dependencies, while the classifier assigns the appropriate label to the sentence. 

The rules of the pipelines are handcrafted and developed by inspecting a dataset composed by comments related to vaccination, collected from a popular parenting forum (BabyCenter.com https://community.babycenter.com/). We share a sample of these comments in this repository.

__________________________
### Requirements

```
python   (3.7.4)

spacy    (2.2.3)
pandas   (0.25.1)
numpy    (1.17.2)
nltk     (3.4.5)
networkx (2.3)
spacy    (2.2.3)
```

To load spacy language model:
```
>>> python -m spacy download en_core_web_sm
```
__________________________

### Structure of the repository

1. ```Schedule_noun_pattern_keywords``` contains the keywords and the corresponding dependency rule used to filter relevant matches retrieved with the "schedule_noun" pattern

2. ```Delay_verbs_pattern_keywords``` contains the keywords and the corresponding dependency rule used to filter relevant matches retrieved with the "delay_verbs" pattern

3. ```data``` contains a sample of the comments related to vaccination collected from BabyCenter.com

4. ```results``` contains the outputs of the extraction pipelines: the structure representation of the sentences matched and the final classification of comments

* ```Dependency_tree_functions.py``` contains the scripts to represent the dependency parser of a sentence trough a network (using the networkx library). In addition, there are functions to search information by naviganting the dependency tree

* ```Schedule_pipeline_functions.py``` contains the scripts defining the vaccination scheduling pipeline

* ```text_elaboration.py``` contains the scripts for basic text preprocessing 

* ```Vaccination_schedule_comment_classification.ipynb``` is the notebook in which the pipeline is applied to the sample of comments located in the ```data``` folder


