from setuptools import setup, find_packages
from codecs import open
import os


# for using through the CNES Bastion as an external user, one shal set the env
# var HAL_UID to its HAL username (or gitlab.cnes.fr one if somehow different)
hal_uid = os.getenv('HAL_UID')
gitlab_auth = ''
_ssh = ''
if hal_uid:
    gitlab_auth = 'gu={}@'.format(hal_uid)
    _ssh = '-ssh'

requirements = ['numpy',
                'mc-cnn @ git+ssh://{}git@gitlab{}.cnes.fr/OutilsCommuns/CorrelateurChaine3D/mc-cnn.git@master'.format(
                    gitlab_auth, _ssh),
                'pandora @ git+ssh://{}git@gitlab{}.cnes.fr/OutilsCommuns/CorrelateurChaine3D/pandora.git@master'.format(
                    gitlab_auth, _ssh)]

def readme():
    with open("README.md", "r", "utf-8") as f:
        return f.read()


setup(name='plugin_mc_cnn',
      version_format='{sha}',
      setup_requires=['very-good-setuptools-git-version'],
      description='Pandora plugin to create the cost volume with the neural network mc-cnn',
      long_description=readme(),
      packages=find_packages(),
      install_requires=requirements,
      entry_points="""
          [pandora.plugin]
          plugin_mc_cnn = plugin_mc_cnn.plugin_mc_cnn:MCCNN
      """,
      )
