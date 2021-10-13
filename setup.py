from setuptools import setup


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
    "networkx>=2.6.3",
    "PyGeometry-z6>=2.0",
    "zuper-commons-z6>=6.1.5",
    "compmake-z6>=6.0.8,<7",
    "quickapp-z6>=6.0.5,<7",
    "reprep-z6>=6.0.3,<7",
    "zuper-typing-z6>=6.1.0",
]

module = "dg_commons"
package = "dg-commons"
src = "src"

version = get_version(filename=f"src/{module}/__init__.py")

setup(
    name=package,
    package_dir={"": src},
    packages=[module, "sim"],
    version=version,
    zip_safe=False,
    install_requires=install_requires,
)
