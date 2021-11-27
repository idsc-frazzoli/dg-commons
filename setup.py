import distutils.text_file
from pathlib import Path
from typing import List

from setuptools import setup, find_packages


def get_version(filename):
    import ast

    version = None
    with open(filename) as f:
        for line in f:
            if line.startswith("__version__"):
                version = ast.parse(line).body[0].value.s
                break
        else:
            raise ValueError("No version found in %r." % filename)
    if version is None:
        raise ValueError(filename)
    return version


def _parse_requirements(filename: str) -> List[str]:
    """Return requirements from requirements file."""
    return distutils.text_file.TextFile(filename=str(Path(__file__).with_name(filename))).readlines()


install_requires = _parse_requirements("requirements.txt")
extras_require = {"all": _parse_requirements("requirements-extra.txt")}

module = "dg_commons"
package = "dg-commons"
src = "src"

version = get_version(filename=f"src/{module}/__init__.py")

setup(
    name=package,
    author="Alessandro Zanardi",
    author_email="azanardi@ethz.ch",
    url="https://github.com/idsc-frazzoli/dg-commons",
    description="Common tools and utilities related to Driving Games",
    long_description=(Path(__file__).with_name("README.md")).read_text(),
    long_description_content_type="text/markdown",
    package_dir={"": src},
    packages=find_packages("src"),
    version=version,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    install_requires=install_requires,
    extras_require=extras_require,
)
