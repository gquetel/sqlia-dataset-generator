let
  inputs = import ./npins;
  pkgs = import inputs.nixpkgs {
    config.allowUnfree = true;
    # https://discourse.nixos.org/t/on-nixpkgs-and-the-ai-follow-up-to-2023-nix-developer-dialogues/37087
    # config.cudaSupport = true;
  };

   sqlmap = pkgs.python3Packages.sqlmap.overridePythonAttrs (oldAttrs: {
    propagatedBuildInputs = (oldAttrs.propagatedBuildInputs or [ ]) ++ [
      pkgs.python3Packages.sqlalchemy 
      pkgs.python3Packages.pymysql 
    ];
  });


  mysql-connector =
    let
      pname = "mysql-connector-python";
      version = "9.3.0";
      format = "wheel";
    in
    pkgs.python312.pkgs.buildPythonPackage {
      # Have to use direct fetchurl as package is not updated in nixkpgs
      inherit pname version format;
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/23/1d/8c2c6672094b538f4881f7714e5332fdcddd05a7e196cbc9eb4a9b5e9a45/mysql_connector_python-9.3.0-py2.py3-none-any.whl";
        sha256 = "sha256-irdxnWFM9UY1IQgvq4avwhraUEtTgWYJDgDuqh/3Kbw=";
      };
      doCheck = false;
    };

  pythonEnv = (
    (pkgs.python312.withPackages (
      ps:
      [
        # Required for generation
        ps.pandas
        ps.numpy
        ps.tqdm

        # Used for training / evaluation
        ps.matplotlib
        ps.scikit-learn

        # Notebooks
        ps.ipykernel
        ps.jupyter
        ps.matplotlib-venn
        # Diversity metric + WAFAMOLE loading.
        ps.sqlglot
        ps.sqlparse

        # BERT model
        ps.accelerate
        ps.evaluate
        ps.torch
        ps.transformers
      ]
      ++ [ mysql-connector ]
    )).override
      (args: {
        ignoreCollisions = true;
      })
  );
in
pkgs.mkShell rec {
  packages = [
    pythonEnv
    pkgs.percona-toolkit
    pkgs.mysql84
    pkgs.metasploit
    sqlmap
  ];

  allowUnfree = true;
  catchConflicts = false;
  shellHook = ''
    export CUSTOM_INTERPRETER_PATH="${pythonEnv}/bin/python"

    # The dataset generation requires a MySQL Server running. We start one using the 
    # following code. 
    export MYSQL_HOME="$PWD/.mysql"
    export MYSQL_DATADIR="$MYSQL_HOME/data"
    export MYSQL_UNIX_PORT="$MYSQL_HOME/mysql.sock"
    export MYSQL_PORT=61337

    # Create MySQL directories if they don't exist
    mkdir -p "$MYSQL_DATADIR"
    mkdir -p "$MYSQL_HOME/tmp"
    mkdir -p "$MYSQL_HOME/log"

    # Initialize MySQL database if not already done
    if [ ! -f "$MYSQL_DATADIR/mysql/db.frm" ] && [ ! -d "$MYSQL_DATADIR/mysql" ]; then
      echo "Initializing MySQL database..."
      mysqld --initialize-insecure \
        --datadir="$MYSQL_DATADIR" \
        --basedir="${pkgs.mysql84}" \
        --user=$USER
    fi

    # Function to start MySQL
    start_mysql() {
      if [ -f "$MYSQL_HOME/mysqld.pid" ] && kill -0 $(cat "$MYSQL_HOME/mysqld.pid") 2>/dev/null; then
        echo "MySQL is already running on port $MYSQL_PORT"
      else
        echo "Starting MySQL on port $MYSQL_PORT..."
        mysqld \
          --datadir="$MYSQL_DATADIR" \
          --socket="$MYSQL_UNIX_PORT" \
          --port=$MYSQL_PORT &
        echo $! > "$MYSQL_HOME/mysqld.pid"

        # Wait for MySQL to start
        for i in {1..30}; do
          if mysqladmin --socket="$MYSQL_UNIX_PORT" ping &>/dev/null; then
            echo "MySQL started successfully!"

            # Initialize databases with bootstrap.sql
            # Change to data directory so SOURCE commands work with relative paths
            echo "Running bootstrap.sql..."
            (cd data && mysql --socket="$MYSQL_UNIX_PORT" -u root < bootstrap.sql)
            echo "Databases initialized!"

            return 0
          fi
          sleep 1
        done
        echo "Failed to start MySQL"
        return 1
      fi
    }

    # Function to stop MySQL
    stop_mysql() {
      if [ -f "$MYSQL_HOME/mysqld.pid" ]; then
        echo "Stopping MySQL..."
        kill $(cat "$MYSQL_HOME/mysqld.pid") 2>/dev/null || true
        rm -f "$MYSQL_HOME/mysqld.pid"
        echo "MySQL stopped"
      else
        echo "MySQL is not running"
      fi
    }

    # Function to restart MySQL
    restart_mysql() {
      stop_mysql
      sleep 2
      start_mysql
    }

    # Export functions
    export -f start_mysql
    export -f stop_mysql
    export -f restart_mysql

    # Auto-start MySQL when entering the shell
    start_mysql

    echo ""
    echo "MySQL development environment ready!"
    echo "  Port: $MYSQL_PORT"
    echo "  Socket: $MYSQL_UNIX_PORT"

    # List all databases (excluding system databases)
    DATABASES=$(mysql --socket="$MYSQL_UNIX_PORT" -u root -N -e "SHOW DATABASES;" 2>/dev/null | grep -vE '^(information_schema|performance_schema|mysql|sys)$' | tr '\n' ', ' | sed 's/, $//')
    if [ -n "$DATABASES" ]; then
      echo "  Databases: $DATABASES"
    else
      echo "  Databases: (none)"
    fi

    echo "  User: tata / Password: tata"
    echo "  Root user: root / Password: root"
    echo ""
    echo "Available commands:"
    echo "  start_mysql   - Start MySQL server"
    echo "  stop_mysql    - Stop MySQL server"
    echo "  restart_mysql - Restart MySQL server"
    echo ""
    echo "To connect: mysql --socket=$MYSQL_UNIX_PORT -u tata -ptata"
    echo ""
  '';
}
