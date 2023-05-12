from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pygof",
    version="0.0.0",
    description="A simple Python Module for Goodness-of-fit test",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    author="aletgn",
    author_email="not_provided",
    url="https://github.com/aletgn/chi2_test",
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.6",
    install_requires=["numpy", "scipy", "matplotlib"],
    extras_require={"test" : ["notebook"],
                    "dev" : ["pytest", "twine"]}
)
