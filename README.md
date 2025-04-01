# Queries Generator for MySQL IDS Testing

This project is designed to generate both normal SQL queries and queries containing a SQL Injection attack. Queries are constructed using manually made templates, dictionnaries and SQL Injection payloads extracted from [sqlmap](https://github.com/sqlmapproject/sqlmap/tree/1.8.7) (version 1.8.7). The generator can use 3 different lexical domains / database schemas based on real-world data to construct the dataset:

- Airbnb Paris	https://insideairbnb.com/get-the-data/
- Airport https://ourairports.com/data/
- UCL https://www.kaggle.com/datasets/hammadjavaid/ucl-matches-and-players-data-20222023

## Usage

The program accepts one parameter, `-ini`, which should point to a `.ini` file containing several configuration sections described below. No external package is required, you can simply run the following command to generate a dataset. 

```
python3 ./main.py -ini ini.ini 
```