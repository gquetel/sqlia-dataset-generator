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

      inherit pname version format;
      src = pkgs.fetchPypi {
        inherit pname version;
        sha256 = "sha256-ixbVFEfjYD8YR4+1oZszO/tz+1j4cusFWhBWNfU9I0U="; 
      };
      doCheck = false;
    };

  pythonEnv = (
    (pkgs.python311.withPackages (
      ps:
      [
        ps.pandas
        ps.numpy
        ps.tqdm
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
    pkgs.sqlmap
  ];

  catchConflicts = false;
  shellHook = ''
    export CUSTOM_INTERPRETER_PATH="${pythonEnv}/bin/python"
  '';
}
