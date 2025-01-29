from setuptools import setup, find_packages

setup(
    name="dsn_fpl",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "networkx",
        "requests",
        "aiohttp"
    ]
) 