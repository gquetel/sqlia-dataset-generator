let
  inputs = import ./npins;
  pkgs = import inputs.nixpkgs { };

  pythonEnv = (
    (pkgs.python311.withPackages (ps: [
      ps.pandas
      ps.numpy
      ps.mysql-connector

    ])).override
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
