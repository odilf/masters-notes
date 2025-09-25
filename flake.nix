{
  description = "Notes for my Master's course on Computational and Applied Mathematics at UC3M";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in

      {
        devShells = {
          default = pkgs.mkShell {
            packages = [
              pkgs.marksman
              pkgs.typos-lsp

              pkgs.typst
              pkgs.tinymist
              pkgs.typstyle
              pkgs.newcomputermodern

              pkgs.jujutsu
              pkgs.just

              pkgs.pnpm_9
              pkgs.nodejs_24
            ];
          };

          ci = pkgs.mkShell {
            packages = [
              pkgs.typst
              pkgs.just
            ];
          };
        };

        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
