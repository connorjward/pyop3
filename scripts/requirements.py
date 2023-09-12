import sys

import toml

pyproject = toml.load("pyproject.toml")
if "build" in sys.argv:
    packages = pyproject["build-system"]["requires"]
elif "requirements" in sys.argv:
    packages = pyproject["project"]["dependencies"]

print("\n".join([pkg for pkg in packages if pkg != "petsc4py"]))
