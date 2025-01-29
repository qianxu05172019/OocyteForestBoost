from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="oocyteforestboost",
    version="0.1.0",
    author="Qian Xu",
    author_email="qianxu0517@gmail.com",
    description="A machine learning framework for predicting cell-cell interactions during oocyte maturation using Random Forest and XGBoost",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qianxu05172019/OocyteForestBoost",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "xgboost>=1.4.2",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "seaborn>=0.11.2",
        "matplotlib>=3.4.2",
    ],
)
