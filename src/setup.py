from setuptools import find_packages, setup

setup(
    name="sunycell", 
    packages=find_packages(),
    description="Toolkit for working with the SUNYCell image repository.",
    license='Apache Software License 2.0',
    version='0.1.1',
    install_requires=[
        'histomicstk',
        'python-dotenv',
        'pooch',
        'rasterio'
    ],
)

