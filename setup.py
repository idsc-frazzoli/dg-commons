import pathlib

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


install_requires = [
    "frozendict>=2.0.6",
    "cytoolz>=0.11.0",
    "cachetools~=4.2.1",
    "numpy>=1.21.2",
    "scipy>=1.7.1",
    "matplotlib==3.4.3",
    "shapely>=1.7.0",
    "PyGeometry-z6>=2.0",
    "zuper-commons-z6>=6.1.5",
]

module = "dg_commons"
package = "dg-commons"
src = "src"

version = get_version(filename=f"src/{module}/__init__.py")

setup(
    name=package,
    author="Alessandro Zanardi",
    author_email="azanardi@ethz.ch",
    url="https://github.com/idsc-frazzoli/dg-commons",
    description='Common tools and utilities related to Driving Games',
    long_description=(pathlib.Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    package_dir={"": src},
    packages=find_packages("src"),
    version=version,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    install_requires=install_requires,
)
