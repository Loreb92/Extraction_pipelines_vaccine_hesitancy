# Extraction pipelines vaccine hesitancy
### Repository for the Lagrange Scholarships Projects about Vaccine Hesitancy - Extraction pipelines  

In this project, we developed two high precision rule-based extraction pipelines able to classify users with respect to vaccination behaviors and experiences from user-generated content. The items we tracked are (i) adherence to the recommended or alternative vaccination schedule and (ii) mentions of positive or negative experiences with adverse events following immunization (AEFI).  

The two pipelines share the same workflow and work at the level of sentences. They are made up by a filter and a classifier. The filter identifies sentences which contain information relevant to the item under consideration by using a combination of rules based on the occurrence of certain keyword with specific syntactic dependencies, while the classifier assigns the appropriate label to the sentence. To classify users, the labels of their comments are propagated. 

The rules of the pipelines are handcrafted and developed by inspecting a dataset composed by comments related to vaccination, collected from a popular parenting forum (BabyCenter.com https://community.babycenter.com/). We share a sample of these comments in this repository.

__________________________
### Requirements

```
spacy
pandas

```
__________________________

### Structure of the repository

1. ```code```

* ```vaccination_schedule``` contains the scripts to retrieve comments related to the vaccination schedule of the author and classify them depending on whether the schedule is the "recommended" or "alternative", and the classification of users based on their schedule-related comments.

* ```experiences_AEFI``` contains the scripts to retrieve comments related to past experiences with AEFI and classify them in "positive_experience" or "negative_experience", and the classification of users based on their comments related to experiences of AEFI.


2. ```data``` contains a sample of the comments related to vaccination collected from BabyCenter.com


3. ```results``` contains the outputs of the extraction pipelines, consisting of the structure representation of the sentences of interest, the final classification of comments and users.


__________________________

### Getting started
