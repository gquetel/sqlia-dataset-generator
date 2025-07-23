# SuperviZ25-SQL

By: 
- Grégor Quétel, mail: gregor.quetel@telecom-paris.fr
- Pierre-François Gimenez, mail: pierre-francois.gimenez@inria.fr
- Thomas Robert, mail: thomas.robert@telecom-paris.fr
- Laurent Pautet, mail: laurent.pautet@telecom-paris.fr
- Eric Alata, mail: ealata@laas.fr

Organisations: PEPR Superviz, Télécom Paris, INRIA, LAAS CNRS.

SuperviZ25-SQL has been designed to evaluate unsupervised SQL Injection detection techniques. The dataset allows to evaluate the capability of mechanisms trained on the benign workload of a unique database schema to detect SQL Injection attack on this targeting this specific deployment. What follows below is the [datasheet](https://arxiv.org/abs/1803.09010) describing this data. 

## Motivation

1. **For what purpose was the dataset created?** *(Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.)*  
    SuperviZ25-SQL has been designed to provide a realist, diverse, properly documented dataset for unsupervised SQL Injection detection. 


2. **Who created this dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?**  
    The dataset was created by members of the PEPR SuperviZ project. 


3. **Who funded the creation of the dataset?** *(If there is an associated grant, please provide the name of the grantor and the grant name and number.)*  
   This work has been partially supported by the French National Research Agency under the France 2030 label (Superviz ANR-22-PECY-0008). 

4. **Any other comments?**  
    None.

## Composition


1. **What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?** *(Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.)*  
   The instances in this dataset represents SQL queries executed against a MySQL database. There are two type of instance: synthetic benign SQL queries and malicious queries containing SQL Injection payloads generated using `sqlmap`.


2. **How many instances are there in total (of each type, if appropriate)?**  
    The dataset contains a total of 3,687,888 instances:
    - Training set: 335,306 benign queries only
    - Test set: 3,352,582 total instances
      - 3,017,390 benign queries (90%)
      - 335,192 malicious queries (10%)
   


1. **Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?** *(If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).)*  
   This dataset is a synthetic sample of the possible instances of SQL queries that are virtually unlimited. However, we tried to provide a realistic coverage of both attacks and normal queries given the query templates. The dataset is based on:
   - A specific database schema from the [OurAirports](https://ourairports.com) project. 
   - 62 manually designed SQL query templates.
    - Attack campaigns targeting a randomly selected subset of 45 out of 62 templates (representing scenarios where only some endpoints are reachable).
    - The benign workload is designed to be representative of real-world SQL traffic patterns, with statement type distributions as follows: 70% SELECT, 10% UPDATE, 10% INSERT, 9% DELETE, 1% administrative.
     
2. **What data does each instance consist of?** *(``Raw'' data (e.g., unprocessed text or images)or features? In either case, please provide a description.)*  
   
Each sample is characterized by the following columns: 
- `full_query`: The full SQL statement.
- `label`: 0 for normal samples, 1 for attacks.
- `user_inputs`: The user input without the query template.
- `attack_stage`: Empty for normal samples, either "recon" or "exploit" for attacks.
- `tamper_method`: Empty for normal samples, for attacks this field designate the randomly selected tamper script used by `sqlmap` to mutate this sample. 
- `attack_status`: Empty for normal samples, else this indicates whether the attack campaign from which this sample was generated, succeeded ("success") or not ("failure"). 
- `statement_type`: Either "select", "delete", "execute", "modify" or "admin".
- `query_template_id`: The query template ID associated to this sample. 
- `attack_id`: Empty for normal samples, else the attack campaign identifier. 
- `attack_technique`: Empty for normal samples, else the `sqlmap` technique used for this attack: "boolean", "error", "inline", "stacked", "time" or "union".
- `split`: Proposed split for this dataset: either "train" or "test". 
   

1. **Is there a label or target associated with each instance? If so, please provide a description.**  
   Yes, each instance has a binary label: either 0 = benign or 1 = malicious. Additionally, malicious instances have fine-grained labels indicating the specific attack technique used (boolean-based, error-based, time-based, union-based, stacked queries, or inline queries).


1. **Is any information missing from individual instances?** *(If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.)*  
   No.
    

7. **Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)?** *(If so, please describe how these relationships are made explicit.)*  
   Instances can be related by the SQL query template they originate from or the attack campaign they are part of (given by the `attack_id` column). 
    
8. **Are there recommended data splits (e.g., training, development/validation, testing)?** *(If so, please provide a description of these splits, explaining the rationale behind them.)*  
   Yes, a training and a testing split is provided through the `split` column. The test split was generated first: 335,192 attacks were collected. We had an objective to provide an imbalanced test set comprising 10% attacks samples, hence we generated 3,017,390 benign queries. For the train set, we choose to provide a set of instance of roughly the same size to the attack set, leading to a train set with 335,306 instances.

1. **Are there any errors, sources of noise, or redundancies in the dataset?** *(If so, please provide a description.)*  
    We cannot be sure that all samples generated with `sqlmap` actually contains a malicious payload, for instance the tool sometimes send queries without payloads to test the HTTP connection. Hence, we tried to minimize labeling errors of attack samples by removing all contradictory instances: instances with the same `full_query` value but with different `label` value.

    Redundancies in SQL queries is common, since we aimed at providing a realistic dataset, such behavior can be observed in our dataset.    


10. **Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?** *(If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.)*  
    The dataset is self-contained.
    


11. **Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals' non-public communications)?** *(If so, please provide a description.)*  
    No.


12. **Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** *(If so, please describe why.)*  
    No.
 
13. **Any other comments?**  
    None.

## Collection Process


1. **How was the data associated with each instance acquired?** *(Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.)*  
    Each instance was synthetically generated. 


1. **What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?** *(How were these mechanisms or procedures validated?)*  
   Instances were generated using our publicly available [generator](https://github.com/gquetel/sqlia-dataset). Normal samples were syntactically validated by submitting them to a MySQL server. No validation on the instances built using `sqlmap` were performed.    


1. **If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?**  
    Not applicable.


1. **Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?**  
    Authors took part in the design of the generator. They were paid through their employment at their respective institutions.


1. **Over what timeframe was the data collected?** *(Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)?  If not, please describe the timeframe in which the data associated with the instances was created.)*  
    The dataset generation roughly took 10 hours.


1. **Were any ethical review processes conducted (e.g., by an institutional review board)?** *(If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.)*  
    No.
    

## Preprocessing/cleaning/labeling


1. **Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?** *(If so, please provide a description. If not, you may skip the remainder of the questions in this section.)*  
    8233 contradictory instances were removed. They consisted of instances found with both labels, originating from `sqlmap` that sometimes sends queries without an actual malicious payload. 


1. **Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?** *(If so, please provide a link or other access point to the "raw" data.)*  
   No.
    

1. **Is the software used to preprocess/clean/label the instances available?** *(If so, please provide a link or other access point.)*  
    Yes, that code is available in the [generator repository](https://github.com/gquetel/sqlia-dataset).


1. **Any other comments?**  
   None.
    


## Uses


1. **Has the dataset been used for any tasks already?** *(If so, please provide a description.)*   

    Yes, 9 unsupervised SQL Injection detection pipelines were trained and tested on the dataset. Each pipeline consist of a feature extraction mechanism: 
    - CountVectorizer: A word count based approach, provided by scikit-learn. 
    - Manually Selected Features, specific to SQL Injection detection, derived from the paper ["A SQL Injection Detection Method Based on Adaptive Deep Forest"](https://doi.org/10.1109/ACCESS.2019.2944951).
    - Sentence-BERT: A pre-trained on cybersecurity data BERT model used to compute a fixed-size vector representation of the query presented in ["SecureBERT: A Domain-Specific Language Model for Cybersecurity"](https://doi.org/10.1007/978-3-031-25538-0_3).

    Combined with one of the following novelty detectors: 
    - Autoencoder: Implemented using PyTorch.
    - Local Outlier Factor: Using the scikit-learn implementation.
    - One-Class Support Vector Machines: Using the scikit-learn implementation.
    
    All implementations are available in the [generator repository](https://github.com/gquetel/sqlia-dataset).

2. **Is there a repository that links to any or all papers or systems that use the dataset?** *(If so, please provide a link or other access point.)*

    No.


3. **What (other) tasks could the dataset be used for?**
    
    The dataset provides a train and test split which does not allow the usage of supervised SQL Injection detection, however with correct preprocessing, one could derive a supervised dataset from the provided instances. 

4. **Are there tasks for which the dataset should not be used?** *(If so, please provide a description.)*
    
    No.

5. **Any other comments?**
    
    None.

## Distribution


1. **Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?** *(If so, please provide a description.)*
    
    Yes, the dataset available under the MIT License.


1. **How will the dataset will be distributed (e.g., tarball  on website, API, GitHub)?** *(Does the dataset have a digital object identifier (DOI)?)*
    
    The dataset is available in a CSV format on the [Zenodo](https://zenodo.org/records/15744477) platform.


1. **When will the dataset be distributed?**
    
   September 2025.
    


1. **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** *(If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.)*  
    Yes, the dataset is distributed under the MIT license.
    


2. **Have any third parties imposed IP-based or other restrictions on the data associated with the instances?** *(If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.)*  
    No.


1. **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?** *(If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.)*    
    No.


1. **Any other comments?**   
    None.


## Maintenance


1. **Who is supporting/hosting/maintaining the dataset?**  
    Grégor Quetel is maintaining the dataset. The dataset is hosted on the Zenodo platform.


1. **How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**  
    
    E-mail addresses are at the top of this document.


1. **Is there an erratum?** *(If so, please provide a link or other access point.)*  
    
    Currently, no. As errors are encountered, future versions of the dataset may be released (but will be versioned). They will all be provided in the same Zenodo location.


1. **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances')?** *(If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?)*  

    Same as previous.


1. **Will older versions of the dataset continue to be supported/hosted/maintained?** *(If so, please describe how. If not, please describe how its obsolescence will be communicated to users.)*  
    
    Yes; all data will be versioned.


1. **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** *(If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.)*  
    No.    


1. **Any other comments?**
    None.
    


