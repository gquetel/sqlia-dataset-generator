-- Bootstrap script to set up databases and user
-- Execute as root
-- Each dataset gets its own database
ALTER USER 'root' @'localhost' IDENTIFIED BY 'root';

DROP USER IF EXISTS 'tata'@'localhost';
FLUSH PRIVILEGES;

CREATE USER 'tata'@'localhost' IDENTIFIED BY 'tata';
FLUSH PRIVILEGES;

-- Load dataset-specific schemas (each creates its own database)
SOURCE ./databases/airport/init_db.sql;