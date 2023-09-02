import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="test_UniP",
    version="0.0.1",
    author="Haocheng Zhao",
    author_email="Haocheng.Zhao@hotmail.com",
    description="A test package for UniP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nobreakfast/test_UniP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    python_requires=">=3.6",
)
