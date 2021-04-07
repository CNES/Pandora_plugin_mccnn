"""
This module contains the required libraries and softwares allowing to execute the software,
and setup elements to configure and identify the software.
"""

from codecs import open as copen
from setuptools import setup, find_packages


requirements = ['numpy',
                'mc-cnn==0.0.1',
                'pandora==1.0.*',
                'nose2']


def readme():
    with copen('README.md', 'r', 'utf-8') as fstream:
        return fstream.read()


setup(name='pandora_plugin_mc_cnn',
      version='x.y.z',
      description='Pandora plugin to create the cost volume with the neural network mc-cnn',
      long_description=readme(),
      packages=find_packages(),
      install_requires=requirements,
      entry_points="""
          [pandora.plugin]
          pandora_plugin_mc_cnn = pandora_plugin_mc_cnn.plugin_mc_cnn:MCCNN
      """,
      include_package_data=True,
      )
