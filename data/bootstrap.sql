-- Bootstrap script to set up databases and user
-- Execute as root
-- Each dataset gets its own database
ALTER USER 'root' @'localhost' IDENTIFIED BY 'root';

DROP USER IF EXISTS 'tata'@'localhost';
FLUSH PRIVILEGES;

CREATE USER 'tata'@'localhost' IDENTIFIED BY 'tata';
FLUSH PRIVILEGES;

-- Dataset-specific schemas are created on-demand during generation
-- (init_dataset_db / stop_mysql_server in db_cnt_manager.py)
-- to ensure each dataset is isolated from the others during sqlmap runs.