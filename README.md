# Queries Generator for MySQL IDS Testing

This project is designed to generate both normal SQL queries and queries containing a SQL Injection attack. Queries are constructed using manually made templates, dictionaries and SQL Injection payloads extracted from [sqlmap](https://github.com/sqlmapproject/sqlmap/tree/1.8.7) (version 1.8.7). The generator can use the following real-world schemas to construct the dataset from:

- airport https://ourairports.com/data/, schema description: https://ourairports.com/help/data-dictionary.html
- UCL https://www.kaggle.com/datasets/hammadjavaid/ucl-matches-and-players-data-20222023

## Usage

The program accepts one parameter, `-ini`, which should point to a `.ini` file containing several configuration sections described below. 

```
python3 ./main.py -ini ini.ini 
```

```
sqlmap -v 3  --skip-waf -D dataset --level=5 --risk=1 --batch --skip='user-agent,referer,host'  --eval="import random;airports_wikipedia_link=random.choice(['https://fr.wikipedia.org/wiki/A%C3%A9rodrome_de_Bagnoles-de-l%27Orne_-_Couterne', 'https://en.wikipedia.org/wiki/Funter_Bay_Seaplane_Base', 'https://en.wikipedia.org/wiki/Dalbandin_Airport', 'https://en.wikipedia.org/wiki/Cessna_Aircraft_Field', 'https://en.wikipedia.org/wiki/Montgomery_County_Airpark', 'https://en.wikipedia.org/wiki/Khwai_River_Airport', 'https://en.wikipedia.org/wiki/Magan_Airport', 'https://en.wikipedia.org/wiki/Tiksi_North', 'https://en.wikipedia.org/wiki/Barrie-Orillia_(Lake_Simcoe_Regional)_Airport', 'https://en.wikipedia.org/wiki/Chehalis%E2%80%93Centralia_Airport']);"   -p 'airports_ident'  -tamper="randomcase" --technique=T -u "http://localhost:8080/airport-U3?airports_wikipedia_link=https%3A%2F%2Fde.wikipedia.org%2Fwiki%2FFlugplatz_Schw%25C3%25A4bisch_Hall-Hessental&airports_ident=30XA" 
```

```
CREATE TABLE airport (
    id INT PRIMARY KEY AUTO_INCREMENT,
    ident VARCHAR(10) NOT NULL UNIQUE,
    type VARCHAR(20) NOT NULL,
    name VARCHAR(255) NOT NULL,
    latitude_deg DECIMAL(10, 6),
    longitude_deg DECIMAL(10, 6),
    elevation_ft INT,
    continent CHAR(2),
    iso_country CHAR(2),
    iso_region VARCHAR(10),
    municipality VARCHAR(100),
    scheduled_service VARCHAR(3),
    gps_code VARCHAR(10), 
    icao_code VARCHAR(10),
    iata_code VARCHAR(3),
    local_code VARCHAR(10),
    home_link VARCHAR(255),
    wikipedia_link VARCHAR(255),
    keywords TEXT
);

CREATE TABLE airport_frequencies (
    id INTEGER PRIMARY KEY,
    airport_ref INTEGER,
    airport_ident VARCHAR(10),
    type VARCHAR(10),
    description VARCHAR(255),
    frequency_mhz DECIMAL(6,3),
    FOREIGN KEY (airport_ref) REFERENCES airport(id),
    FOREIGN KEY (airport_ident) REFERENCES airport(ident)
);

CREATE TABLE runways (
    id INT PRIMARY KEY AUTO_INCREMENT,
    airport_ref INT NOT NULL,
    airport_ident VARCHAR(10) NOT NULL,
    length_ft INT,
    width_ft INT,
    surface VARCHAR(20),
    lighted TINYINT(1),
    closed TINYINT(1),
    le_ident VARCHAR(10),
    le_latitude_deg DECIMAL(10, 6),
    le_longitude_deg DECIMAL(10, 6),
    le_elevation_ft INT,
    le_heading_degT DECIMAL(5, 1),
    le_displaced_threshold_ft INT,
    he_ident VARCHAR(10),
    he_latitude_deg DECIMAL(10, 6),
    he_longitude_deg DECIMAL(10, 6),
    he_elevation_ft INT,
    he_heading_degT DECIMAL(5, 1),
    he_displaced_threshold_ft INT,
    FOREIGN KEY (airport_ref) REFERENCES airport(id),
    FOREIGN KEY (airport_ident) REFERENCES airport(ident)
);

CREATE TABLE navaids (
    id INTEGER PRIMARY KEY,
    filename VARCHAR(255),
    ident VARCHAR(3),
    name VARCHAR(255),
    type VARCHAR(10),
    frequency_khz INTEGER,
    latitude_deg DECIMAL(10,6),
    longitude_deg DECIMAL(10,6),
    elevation_ft INTEGER,
    iso_country VARCHAR(2),
    dme_frequency_khz INTEGER,
    dme_channel VARCHAR(10),
    dme_latitude_deg DECIMAL(10,6),
    dme_longitude_deg DECIMAL(10,6),
    dme_elevation_ft INTEGER,
    slaved_variation_deg DECIMAL(7,3),
    magnetic_variation_deg DECIMAL(7,3),
    usageType VARCHAR(10),
    power VARCHAR(10),
    associated_airport VARCHAR(10),
    FOREIGN KEY (associated_airport) REFERENCES airport(ident)
);

CREATE TABLE countries (
    id INTEGER PRIMARY KEY,
    code VARCHAR(2) UNIQUE NOT NULL,
    name VARCHAR(100),
    continent VARCHAR(2),
    wikipedia_link VARCHAR(500),
    keywords TEXT
);

CREATE TABLE regions (
    id INTEGER PRIMARY KEY,
    code VARCHAR(10) UNIQUE NOT NULL,
    local_code VARCHAR(10),
    name VARCHAR(100),
    continent VARCHAR(2),
    iso_country VARCHAR(2),
    wikipedia_link VARCHAR(500),
    keywords TEXT
);
```