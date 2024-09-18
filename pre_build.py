import subprocess
import sys
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        print("CustomBuildHook initialized")
        try:
            print("Compiling EpiSim...")
            subprocess.run(["bash", "install.sh"], check=True)
            print("Compilation successful")
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e}", file=sys.stderr)
            sys.exit(1)

    def clean(self, versions):
        print("CustomBuildHook clean method called")
