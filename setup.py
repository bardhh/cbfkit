from setuptools import setup, find_packages
import sys, platform

# Default dependencies
install_requires = ["numpy", "scipy", "matplotlib", "sympy","control"]

# OS-specific dependencies
# import conditional if system is mac m1
if platform.system() == "Darwin" and platform.machine() == "arm64":
    # this is a 64-bit OS X system
    install_requires.append("kvxopt")
else:
    # this is a 32-bit OS X system or a Linux system
    install_requires.append("cvxopt")
    #! Note: ROS needs to be installed separately

# Define common arguments to the setup() function
common_args = dict(
    name="CBFkit",
    version="0.1",
    packages=[
        "cbfkit",
        "cbfkit.models",
        "cbfkit.tutorial",
    ],
    package_dir={"": "src", "cbfkit.models": "src/cbfkit/models", "cbfkit.tutorial": "src/cbfkit/tutorial"},
    author="Bardh Hoxha, Mitchell Black, Hideki Okamoto, Georgios Fainekos",
    author_email="bardhh@gmail.com",
    description="CBFkit is a python package for control barrier functions",
    license="MIT",
    keywords="control barrier functions, ROS, robotics, safety filter",
    long_description_content_type="text/markdown",
    url="TBA",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Control Theory",
    ],
)

# Combine the common arguments with the dependencies and call the setup() function
setup(install_requires=install_requires, **common_args)
