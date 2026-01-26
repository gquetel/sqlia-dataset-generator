let
  inputs = import ./npins;
  pkgs = import inputs.nixpkgs {
    config.allowUnfree = true;
    # https://discourse.nixos.org/t/on-nixpkgs-and-the-ai-follow-up-to-2023-nix-developer-dialogues/37087
    # config.cudaSupport = true;
  };

  sqlmap = pkgs.python313Packages.sqlmap.overridePythonAttrs (oldAttrs: {
    propagatedBuildInputs = (oldAttrs.propagatedBuildInputs or [ ]) ++ [
      pkgs.python313Packages.sqlalchemy
      pkgs.python313Packages.pymysql
    ];
  });
  # Custom version 9.0.2 with built-in subtests. Can use packaged if 
  # version is superior to 9.0.0 which the one where they merged subtest into core.
  pytest = pkgs.python313Packages.pytest.overridePythonAttrs (oldAttrs: rec {
    version = "9.0.2";
    src = pkgs.fetchPypi {
      pname = "pytest";
      inherit version;
      sha256 = "sha256-dRhmUakr2JYR0dn8IPC0NF/YJ8QczVwpmoaKBdcO3xE=";
    };
  });

  mysql-connector =
    let
      pname = "mysql-connector-python";
      version = "9.3.0";
      format = "wheel";
    in
    pkgs.python313.pkgs.buildPythonPackage {
      # Have to use direct fetchurl as package is not updated in nixkpgs
      inherit pname version format;
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/23/1d/8c2c6672094b538f4881f7714e5332fdcddd05a7e196cbc9eb4a9b5e9a45/mysql_connector_python-9.3.0-py2.py3-none-any.whl";
        sha256 = "sha256-irdxnWFM9UY1IQgvq4avwhraUEtTgWYJDgDuqh/3Kbw=";
      };
      doCheck = false;
    };

  pythonEnv = (
    (pkgs.python313.withPackages (
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
      ++ [
        mysql-connector
        pytest
      ]
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
    # Formatting tools
    pkgs.treefmt
    pkgs.black
    pkgs.nixpkgs-fmt
    pkgs.taplo
    pkgs.mdformat
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

    # Create MySQL directories
    mkdir -p "$MYSQL_DATADIR"
    mkdir -p "$MYSQL_HOME/tmp"
    mkdir -p "$MYSQL_HOME/log"

    # Initialize MySQL database if not already done
    if [ ! -d "$MYSQL_DATADIR/mysql" ]; then
      echo "Initializing MySQL database..."
      mysqld --initialize-insecure \
        --datadir="$MYSQL_DATADIR" \
        --basedir="${pkgs.mysql84}" \
        --user=$USER
    fi

    # Check if MySQL is already running
    MYSQL_STARTED=false
    MYSQL_ALREADY_RUNNING=false
    if mysqladmin --socket="$MYSQL_UNIX_PORT" ping &>/dev/null; then
      echo "MySQL already running on port $MYSQL_PORT"
      MYSQL_STARTED=true
      MYSQL_ALREADY_RUNNING=true
    else
      # Start MySQL
      echo "Starting MySQL on port $MYSQL_PORT..."
      mysqld \
        --datadir="$MYSQL_DATADIR" \
        --socket="$MYSQL_UNIX_PORT" \
        --port=$MYSQL_PORT &
      MYSQL_PID=$!
      echo $MYSQL_PID > "$MYSQL_HOME/mysqld.pid"

      # Wait for MySQL to start
      for i in {1..30}; do
        if mysqladmin --socket="$MYSQL_UNIX_PORT" ping &>/dev/null; then
          echo "MySQL started successfully!"
          MYSQL_STARTED=true
          break
        fi
        sleep 1
      done
    fi

    if [ "$MYSQL_STARTED" = true ]; then
      # Initialize databases with bootstrap.sql (only if we just started MySQL)
      if [ "$MYSQL_ALREADY_RUNNING" = false ]; then
        # Change to data directory so SOURCE commands work with relative paths
        echo "Running bootstrap.sql..."
        (cd data && mysql --socket="$MYSQL_UNIX_PORT" -u root < bootstrap.sql)
        echo "Databases initialized!"
      fi

      # List all databases (excluding system databases)
      DATABASES=$(mysql --socket="$MYSQL_UNIX_PORT" -u root -proot -N -e "SHOW DATABASES;" 2>/dev/null | grep -vE '^(information_schema|performance_schema|mysql|sys)$' | tr '\n' ', ' | sed 's/, $//')

      echo ""
      echo "MySQL development environment ready!"
      echo "  Port: $MYSQL_PORT"
      echo "  Socket: $MYSQL_UNIX_PORT"
      if [ -n "$DATABASES" ]; then
        echo "  Databases: $DATABASES"
      else
        echo "  Databases: (none)"
      fi
      echo "  User: tata / Password: tata"
      echo "  Root user: root / Password: root"
      echo ""
      echo "To connect: mysql --socket=$MYSQL_UNIX_PORT -u tata -ptata"
      echo "To validate schemas: pytest tests/test_database_schemas.py -v"
    else
      echo "Failed to start MySQL"
    fi

    # Setup cleanup trap for shell exit
    cleanup_mysql() {
      # Only clean up if we started MySQL ourselves
      if [ "$MYSQL_ALREADY_RUNNING" = false ]; then
        if [ -f "$MYSQL_HOME/mysqld.pid" ]; then
          PID=$(cat "$MYSQL_HOME/mysqld.pid")
          echo ""
          echo "Stopping MySQL..."
          kill $PID 2>/dev/null || true

          # Wait for MySQL to fully stop
          for i in {1..30}; do
            if ! kill -0 $PID 2>/dev/null; then
              break
            fi
            sleep 1
          done

          # Force kill if still running
          if kill -0 $PID 2>/dev/null; then
            kill -9 $PID 2>/dev/null || true
            sleep 1
          fi
        fi

        # Clean up MySQL directory
        if [ -d "$MYSQL_HOME" ]; then
          echo "Cleaning up MySQL data..."
          rm -rf "$MYSQL_HOME"
          echo "MySQL cleanup complete"
        fi
      fi
    }

    trap cleanup_mysql EXIT
    echo ""
  '';
}
