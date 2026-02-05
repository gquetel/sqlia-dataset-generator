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
    pkgs.perl # perl is required by pt-kill (missing Sys/Hostname.pm)
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

    # MySQL environment variables used by mysql-start / mysql-stop scripts.
    # Data lives in /tmp (local to each machine) to avoid NFS conflicts.
    export MYSQL_HOME="/tmp/mysql-dev-sqlia"
    export MYSQL_DATADIR="$MYSQL_HOME/data"
    export MYSQL_UNIX_PORT="$MYSQL_HOME/mysql.sock"
    export MYSQL_PORT=61337

    # Put project scripts on PATH
    export PATH="$PWD/scripts:$PATH"

    echo "Run 'mysql-start' to start MySQL, 'mysql-stop' to stop it."
  '';
}
