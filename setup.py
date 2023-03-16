import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="forecast", # Replace with your own username
    version="0.1.0",
    author="Fred Commo - KPMG",
    author_email="fcommo@kpmg.fr",
    description="Multiple models optimzation to solve forecasting problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fredcommo/Forecast",
    # package_dir={"forecast": "forecast"},
    packages=["forecast"],
    # packages=setuptools.find_packages(where="Forecast"),
    install_requires=[
        'matplotlib==3.6.3',
        'numpy==1.24.1',
        'optuna==3.1.0',
        'pandas==1.5.3',
        'scikit_learn==1.2.2',
        'xgboost==1.7.4'
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={'forecast': ['data/*.csv']},
    entry_points={'console_scripts': ['demo=forecast.demo:main']},
    python_requires='>=3.8',
)