from setuptools import setup, find_packages

setup(
    name="music-recommendation-server",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "Flask>=3.0.0",
        "pymongo>=4.6.0",
        "PyJWT>=2.8.0",
        "python-dotenv>=1.0.0",
        "bcrypt>=4.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "torch>=2.0.0"
    ]
) 