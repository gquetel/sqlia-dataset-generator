# Queries Generator for MySQL IDS Testing

This project is designed to generate both normal SQL queries and queries containing a SQL Injection attack. Queries are constructed using manually made templates, dictionnaries and SQL Injection payloads extracted from [sqlmap](https://github.com/sqlmapproject/sqlmap/tree/1.8.7) (version 1.8.7). The generator can use the following real-world schemas to construct the dataset from:

- airport https://ourairports.com/data/, schema description: https://ourairports.com/help/data-dictionary.html
- UCL https://www.kaggle.com/datasets/hammadjavaid/ucl-matches-and-players-data-20222023

## Usage

The program accepts one parameter, `-ini`, which should point to a `.ini` file containing several configuration sections described below. 

```
python3 ./main.py -ini ini.ini 
```