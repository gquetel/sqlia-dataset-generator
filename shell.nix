{
  # On a NixOS machine, to support CUDA. You can run: 
  # nix-shell --arg cudaSupport true
  # This will grant access to CUDA to torch and other packages.
  # TODO: Do the same for Ubuntu machines with access to Nix.
  cudaSupport ? false
,
}:
let
  inputs = import ./npins;
  pkgs = import inputs.nixpkgs {
    config.allowUnfree = true;
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
      rev = "40596e0cc45c6cf295790e264745bca4e21d0afc";
      sha256 = "sha256-sxTqC71h3oCP1dkwj8anrk6C3KSdvSraQPEKYQkb1Nc=";
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
      (if cudaSupport then torch-bin else torch)
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

  vendi-score = pkgs.python313.pkgs.buildPythonPackage {
    pname = "vendi-score";
    version = "0.0.3";
    pyproject = true;
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/7f/4c/ffff6368e4f13a17b8b65df59c868f6a2c8c9feadaaa203e7b6ac5e5f659/vendi-score-0.0.3.tar.gz";
      sha256 = "7b133fd293d63038aea032b2933c68a7040991ee91d9b953fb6b1ede43526c53";
    };
    build-system = [ pkgs.python313Packages.setuptools ];
    propagatedBuildInputs = with pkgs.python313Packages; [ numpy scipy scikit-learn ];
    doCheck = false;
  };

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
        ps.shap

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
        (if cudaSupport then ps.torch-bin else ps.torch)
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
        vendi-score
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
    pkgs.corefonts # Times New Roman (and other MS core fonts) for paper figures
    pkgs.cm_unicode # Computer Modern Unicode fonts (CMU Serif / CMU Sans Serif)
    # Formatting tools
    pkgs.treefmt
    pkgs.black
    pkgs.nixpkgs-fmt
    pkgs.taplo
    pkgs.mdformat
  ] ++ pkgs.lib.optionals cudaSupport [
    pkgs.cudaPackages.cudatoolkit
    pkgs.cudaPackages.cudnn
  ];

  allowUnfree = true;
  catchConflicts = false;

  LD_LIBRARY_PATH = pkgs.lib.optionalString cudaSupport
    (pkgs.lib.makeLibraryPath [
      pkgs.cudaPackages.cudatoolkit
      pkgs.cudaPackages.cudnn
      pkgs.stdenv.cc.cc.lib
    ]);
  shellHook = ''
    export CUSTOM_INTERPRETER_PATH="${pythonEnv}/bin/python"

    # Make corefonts (Times New Roman, etc.) and Computer Modern visible to fontconfig
    export FONTCONFIG_FILE="${pkgs.makeFontsConf { fontDirectories = [ pkgs.corefonts pkgs.cm_unicode ]; }}"

    # MySQL environment variables used by mysql-start / mysql-stop scripts.
    # Data lives in /tmp (local to each machine) to avoid NFS conflicts.
    export MYSQL_HOME="/tmp/mysql-dev-sqlia"
    export MYSQL_DATADIR="$MYSQL_HOME/data"
    export MYSQL_UNIX_PORT="$MYSQL_HOME/mysql.sock"
    export MYSQL_PORT=61337

    # Put project scripts on PATH
    export PATH="$PWD/scripts:$PATH"

    echo "MySQL servers can be manually run using 'mysql-start', 'mysql-stop' to stop it."
  '';
}
