let
  inputs = import ./npins;
  shell-drv = import ./shell.nix;
  pkgs = import inputs.nixpkgs {
    config.allowUnfree = true;
  };

  rawRepo = pkgs.fetchFromGitHub {
    owner = "gquetel";
    repo = "sqlia-dataset-generator";
    # We do not use 'v1.0.0-anubis' which correspond to the code with which the dataset
    # has been generated but a more recent version with bugfixes and features that do
    # not interfere with the generation code (see commit history).
    rev = "f051fd3be8a8c952004fd2ece544112a6f316fb7";
    hash = "sha256-4d3/N4tLkFuFAtpi3Fsnj9j3YK+BnlqdXTNFrEcO6mM=";
  };

  # Hack to have the code in /generator rather than directly to root.
  # There is probably a better way to do this.
  repo = pkgs.runCommand "repo-in-generator-dir" { } ''
    mkdir -p $out/generator
    cp -r ${rawRepo}/* $out/generator/
  '';
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
        ps.scikit-learn
        
        # Used for training / evaluation
        ps.matplotlib
        ps.matplotlib-venn
        ps.scikit-learn

        # Notebooks
        ps.ipykernel
        ps.jupyter

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

pkgs.dockerTools.buildImage {
  name = "sqlia-dataset";
  copyToRoot = pkgs.buildEnv {
    name = "dpackages";
    paths = [
      # CLI tools to include in container
      pkgs.bashInteractive
      pkgs.coreutils
      pkgs.nano

      # Generation dependencies
      pythonEnv
      mysql-connector
      pkgs.sqlmap
      pkgs.percona-toolkit
      pkgs.mysql84
      pkgs.perl # required by percona-toolkit

      # Generation code
      repo
      # Script rendered available to be run to start mysqld:
      # The root password must match the 'root_password' option in ini.ini
      (pkgs.writeTextFile {
        name = "setup-mysql";
        text = ''
          #!${pkgs.runtimeShell}
          mysqld --initialize-insecure --basedir=/usr/local/mysqld_1/ --datadir=/usr/local/mysqld_1/datadir/
          mysqld --basedir=/usr/local/mysqld_1/ --datadir=/usr/local/mysqld_1/datadir/ --socket=/usr/local/mysqld_1/socket --daemonize
          mysql -u root --skip-password --socket=/usr/local/mysqld_1/socket -e "ALTER USER 'root'@'localhost' IDENTIFIED BY 'verysecurepwd'";
          mysql --user=root  --socket=/usr/local/mysqld_1/socket < ./data/init_db.sql --password="verysecurepwd"
        '';
        destination = "/generator/setup-mysql.sh";
        executable = true;
      })
      # Welcome message
      (pkgs.writeTextFile {
        name = "nobody-bashrc";
        destination = "/generator/.bashrc";
        executable = false;
        text = ''
          echo ""
          echo "Welcome to the SuperviZ-SQL25 Dataset Generator Container"
          echo "----------------------------------------------------------"
          echo ""
          echo "Available commands:"
          echo ""
          echo "  ./setup-mysql.sh"
          echo "      Initialize the MySQL server."
          echo ""
          echo "  python3 ./launcher.py -ini ini.ini --testing"
          echo "      Run the generator in test mode."
          echo ""
          echo "  python3 ./launcher.py -ini ini.ini"
          echo "      Generate the full dataset."
          echo ""
          echo "  docker cp <containerId>:/generator/dataset.csv ./dataset.csv"
          echo "      (From the host) Copy the generated dataset to the host machine."
          echo ""
        '';
      })
    ];

    # pathsToLink is the list of subdirectories to be included in container.
    pathsToLink = [
      "/tmp"
      "/bin"
      "/generator"
    ];
  };

  diskSize = 4096;
  buildVMMemorySize = 2048;

  # Container initialization commands called at BUILD TIME.
  # 1. mysqld is not happy to be launched as root which is default user in container.
  #    We create a nobody user and group.
  # 2. We create /usr/local/mysqld_1/ as our basedir and set the correct rights.
  # 3. mysqld requires access to /tmp
  # 4. We create our home directory /generator and the correct access rights too.

  runAsRoot = ''
    #!${pkgs.runtimeShell}
    ${pkgs.dockerTools.shadowSetup}
    groupadd -r nobody
    useradd -r -g nobody nobody
    mkdir -p /usr/local/mysqld_1/
    chmod 777 /tmp
    chmod -R 755 /generator 
    chown nobody:nobody /generator
    chown nobody:nobody /usr/local/mysqld_1/
  '';

  # Container RunConfig Field Descriptions, available fields listed in [1]
  # - [1]: https://github.com/moby/moby/blob/46f7ab808b9504d735d600e259ca0723f76fb164/image/spec/spec.md#container-runconfig-field-descriptions
  config = {
    Cmd = [ "${pkgs.bashInteractive}/bin/bash" ];
    WorkingDir = "/generator";
    User = "nobody:nobody";
    # required by bashrc
    Env = [ "HOME=/generator" ];
  };
}
