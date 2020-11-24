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
pickle
```

To load spacy language model:
```
>>> python -m spacy download en_core_web_sm-2.2.5 --direct
```
__________________________

### Structure of the repository

1. ```Experiences_AEFI``` contains the keywords used to filter sentences relevant to experiences with adverse events following immunization

2. ```Vaccination_schedule``` contains the keywords used to filter sentences relevant to vaccination scheduling

3. ```data``` contains a sample of the comments related to vaccination collected from BabyCenter.com

4. ```output``` contains the results of the two pipelines

5. ```test``` contains a list of sentences and the corresponding dependency trees. It is useful to test if the dependency parser of SpaCy returns the expected parsing

6. ```utils``` contains files useful for the pipelines

* ```AEFI_pipeline_functions.py``` contains the script thad defines the extraction pipeline of experiences of adverse reactions following immunization.

* ```Dependency_tree_functions.py``` contains the scripts to represent the dependency parser of a sentence trough a network (using the networkx library). In addition, there are functions to search information by naviganting the dependency tree

* ```Experiences AEFI : commentclassification.ipynb``` is the notebook in which the pipeline of experiences of adverse reactions following immunization is applied to the sample of comments located in the ```data``` folder

* ```Schedule_pipeline_functions.py``` contains the scripts defining the vaccination scheduling pipeline

* ```Vaccination schedule : comment classification.ipynb``` is the notebook in which the pipeline is applied to the sample of comments located in the ```data``` folder

* ```test_dependency_parsing.ipynb``` is the notebook in which the dependency parser is tested and compared with the expected behavior

* ```text_elaboration.py``` contains the scripts for basic text preprocessing 
