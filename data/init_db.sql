-- Execute as root on database 
DROP DATABASE IF EXISTS dataset;
DROP USER IF EXISTS 'tata'@'localhost';
FLUSH PRIVILEGES;

create database dataset;
use dataset; 

CREATE USER 'tata'@'localhost' IDENTIFIED BY 'tata';
GRANT ALL PRIVILEGES ON dataset.* TO 'tata'@'localhost';
flush privileges; 

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