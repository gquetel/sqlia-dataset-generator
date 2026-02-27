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

  # Kaleido runtime dependency.
  logistro = pkgs.python313.pkgs.buildPythonPackage {
    pname = "logistro";
    version = "2.0.1";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/54/20/6aa79ba3570bddd1bf7e951c6123f806751e58e8cce736bad77b2cf348d7/logistro-2.0.1-py3-none-any.whl";
      sha256 = "sha256-Bv+hJ7n7SsixlyrmsqnX/eV1mL9ZOc1wj0PsW7otMes=";
    };
    doCheck = false;
  };

  # Kaleido runtime dependency.
  choreographer = pkgs.python313.pkgs.buildPythonPackage {
    pname = "choreographer";
    version = "1.2.1";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/b7/9f/d73dfb85d7a5b1a56a99adc50f2074029468168c970ff5daeade4ad819e4/choreographer-1.2.1-py3-none-any.whl";
      sha256 = "sha256-mvU4Xv+jwgTbwzer96x0/YkIztMmoVZF3DHd51cYx34=";
    };
    propagatedBuildInputs = [
      logistro
      pkgs.python313Packages.simplejson
    ];
    doCheck = false;
  };

  gaur-sql-detect = pkgs.python313.pkgs.buildPythonPackage {
    pname = "gaur-sql-detect";
    version = "0.1.0";
    pyproject = true;
    src = pkgs.fetchFromGitHub {
      owner = "gquetel";
      repo = "gaur-sql-detect";
      rev = "35b6db06df38dfdefa62d7f085a5db606204619a";
      sha256 = "sha256-4C7n1NX9m7D19zaywXsK+ppwHmQaFGyTfpyArDkvlxo=";
    };
    build-system = [ pkgs.python313Packages.setuptools ];
    dependencies = with pkgs.python313Packages; [
      pandas
      numpy
      tqdm
      scipy
      scikit-learn
      plotly
      matplotlib
      tabulate
      torch
      transformers
      accelerate
      sentence-transformers
      evaluate
      mysql-connector
      kaleido
      zstandard
    ];
    doCheck = false;
  };

  # TODO: If we keep the llm2vec pipeline, find a way to relax dependencies 
  # llm2vec = pkgs.python313.pkgs.buildPythonPackage {
  #   pname = "llm2vec";
  #   version = "0.2.3";
  #   src = pkgs.fetchurl {
  #     url = "https://files.pythonhosted.org/packages/79/45/4b71b3d3112d7cb17e9e221ef0a2acd35563f206d7d22ddcf13f460c78c6/llm2vec-0.2.3.tar.gz";
  #     sha256 = "sha256-SrdJFHgUfaA/B85U0kVnDHLLCeR9TeDIdS7wCEFtNfw=";
  #   };
  #   pyproject = true;
  #   build-system = [ pkgs.python313Packages.setuptools ];
  #   dependencies = with pkgs.python313Packages; [
  #     numpy
  #     tqdm
  #     torch
  #     peft
  #     transformers
  #     datasets
  #     evaluate
  #     scikit-learn
  #   ];
  #   doCheck = false;
  # };

  # gensim 4.3.3 in nixpkgs is not supported for python3.13; use 4.4.0 wheel directly.
  gensim = pkgs.python313.pkgs.buildPythonPackage {
    pname = "gensim";
    version = "4.4.0";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/b3/b9/ee43ef9c391857232603a9ee281e9c5953f7922d70c98c2296a037d1c0b7/gensim-4.4.0-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl";
      sha256 = "sha256-kDOxiSC3d05o6vrNvYclL/opOC7EZd24i9A24A/IY2U=";
    };
    propagatedBuildInputs = [
      pkgs.python313Packages.numpy
      pkgs.python313Packages.scipy
      pkgs.python313Packages.smart-open
    ];
    doCheck = false;
  };

  # This lib is used by plotly to export figures. It is very cursed, IT WILL REQUIRE
  # GOOGLE CHROME TO EXPORT THE FIGURE!!! I hate it, but I hate manually saving
  # figures more.
  kaleido = pkgs.python313.pkgs.buildPythonPackage {
    pname = "kaleido";
    version = "1.2.0";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/4b/97/f6de8d4af54d6401d6581a686cce3e3e2371a79ba459a449104e026c08bc/kaleido-1.2.0-py3-none-any.whl";
      sha256 = "sha256-wn7YK1Hfa5I9DmVv6sIhNDoNvNL7m8fmsduX9h6aFRM=";
    };
    propagatedBuildInputs = [
      choreographer
      logistro
      pkgs.python313Packages.orjson
      pkgs.python313Packages.packaging
    ];
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
        ps.plotly
        ps.matplotlib-venn

        # Diversity metric + WAFAMOLE loading.
        ps.sqlglot
        ps.sqlparse

        # BERT model
        ps.accelerate
        ps.evaluate
        ps.torch
        ps.transformers
        ps.sentence-transformers

        # gaur-sql-detect dependencies
        ps.scipy
        ps.tabulate

      ]
      ++ [
        mysql-connector
        pytest
        kaleido
        gensim
        gaur-sql-detect
        # llm2vec
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
    pkgs.chromium # Required by kaleido for plotly figure export... #cursed
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
