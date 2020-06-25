from setuptools import setup, find_packages
from codecs import open
import os


requirements = ['numpy',
                'mc-cnn==0.0.1',
                'pandora==0.2.0rc0']


def readme():
    with open("README.md", "r", "utf-8") as f:
        return f.read()


setup(name='plugin_mc_cnn',
      version='x.y.z',
      description='Pandora plugin to create the cost volume with the neural network mc-cnn',
      long_description=readme(),
      setup_requires=['very-good-setuptools-git-version'],
      packages=find_packages(),
      install_requires=requirements,
      entry_points="""
          [pandora.plugin]
          plugin_mc_cnn = plugin_mc_cnn.plugin_mc_cnn:MCCNN
      """,
      )
