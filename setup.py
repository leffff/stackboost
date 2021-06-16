import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="stackboost",
    version="0.1.5",
    author="leffff",
    author_email="levnovitskiy@gmail.com",
    description="Stacked decision trees algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leffff/stackboost",
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=["numpy", "pandas", "torch", "numba"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
