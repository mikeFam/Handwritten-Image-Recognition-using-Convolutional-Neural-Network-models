import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pugn057 and tpha585",
    version="0.1",
    author_1="Sky Nguyen",
    author_2="Michael Pham",
    author_email_1="pugn057@aucklanduni.ac.nz",
    author_email_2="tpha585@aucklanduni.ac.nz",
    description="Image recognition application with 4 different model of CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UOA-CS302-2020/CS302-Python-2020-Group23",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)