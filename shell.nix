let
  inputs = import ./npins;
  pkgs = import inputs.nixpkgs { };

  mysql-connector =
    let
      pname = "mysql-connector-python";
      version = "9.3.0";
      format = "wheel";
    in
    pkgs.python311.pkgs.buildPythonPackage {
      # Have to use direct fetchurl as package is not updated in nixkpgs
      # 9.3.0 or 9.2.0 are not available at https://files.pythonhosted.org/packages/source/m/mysql-connector-python/mysql-connector-python-9.3.0.tar.gz
      # And i couldn't make fetchPypi use the correct wheel address.
      inherit pname version format;
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/23/1d/8c2c6672094b538f4881f7714e5332fdcddd05a7e196cbc9eb4a9b5e9a45/mysql_connector_python-9.3.0-py2.py3-none-any.whl";
        sha256 = "sha256-irdxnWFM9UY1IQgvq4avwhraUEtTgWYJDgDuqh/3Kbw=";
      };
      doCheck = false;
    };

  pythonEnv = (
    (pkgs.python311.withPackages (
      ps:
      [
        ps.pandas
        ps.numpy
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
  ];

  catchConflicts = false;
  shellHook = ''
    export CUSTOM_INTERPRETER_PATH="${pythonEnv}/bin/python"
  '';
}
