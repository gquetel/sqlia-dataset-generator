# Development Environment

This project uses Nix to provide a complete development environment with MySQL.

## Getting Started

Enter the development shell:
```bash
nix-shell
```

The shell will automatically:
- Set up a local MySQL server on port **61337**
- Initialize databases with `data/bootstrap.sql` (which loads dataset-specific schemas)
- Start the MySQL server in the background
- Create dataset-specific databases (e.g., `airport`) with the `tata` user

## MySQL Information

- **Port**: 61337
- **Socket**: `.mysql/mysql.sock`
- **Databases**: Each dataset has its own database (e.g., `airport`, `dataset_2`)
- **User**: `tata` / Password: `tata`
- **Root**: `root` / No password
- **Data directory**: `.mysql/data` 

## Connecting to MySQL

From within the nix-shell:
```bash
# Connect to airport database as tata user
mysql --socket=$MYSQL_UNIX_PORT -u tata -ptata airport

# Connect as root (no database)
mysql --socket=$MYSQL_UNIX_PORT -u root

# List all databases
mysql --socket=$MYSQL_UNIX_PORT -u root -e "SHOW DATABASES"
```

## MySQL Management Commands

Available in the shell:
```bash
start_mysql    # Start MySQL server
stop_mysql     # Stop MySQL server
restart_mysql  # Restart MySQL server
```
To reset the database, stop the server and remove the `.mysql` directory.