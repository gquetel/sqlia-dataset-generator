-- Bootstrap script to set up databases and user
-- Execute as root
-- Each dataset gets its own database
ALTER USER 'root' @'localhost' IDENTIFIED BY 'root';

DROP USER IF EXISTS 'tata'@'localhost';
FLUSH PRIVILEGES;

CREATE USER 'tata'@'localhost' IDENTIFIED BY 'tata';
FLUSH PRIVILEGES;

-- Load dataset-specific schemas (each creates its own database)
-- TODO: I am unsure this is the right location to do this. Maybe I should instead
-- source the correct file at the begining of the generation of a dataset
-- And drop the database at the end of the generation to reduce side effects.
SOURCE ./datasets/OurAirports/init_db.sql;
SOURCE ./datasets/OHR/init_db.sql;
SOURCE ./datasets/sakila/init_db.sql
SOURCE ./datasets/AdventureWorks/init_db.sql