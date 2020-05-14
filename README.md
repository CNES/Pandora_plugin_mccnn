# Plugin_MC-CNN

Plugin MC-CNN : il permet d'intégrer le réseau mc-cnn dans Pandora.

## Installation pour les utilisateurs sur le cluster HAL du CNES

Cette procédure vous permet d'installer les dépôts pandora et mc-cnn sans les cloner au préalable. Notez que les sources
ne seront pas accessibles avec cette procédure.

```sh
u@m $ module purge
u@m $ module load python/3.7.2 gdal/2.1.1
u@m $ virtualenv myEnv --no-site-packages
u@m $ source myEnv/bin/activate
u@m $ pip install --upgrade pip
(myEnv) u@m $ git clone git@gitlab.cnes.fr:OutilsCommuns/CorrelateurChaine3D/pandora_plugins/plugin_mc-cnn.git
(myEnv) u@m $ pip install -e plugin_mc-cnn
```

## Utilisation

L'utilisation du réseau mc-cnn se fait via [pandora](https://gitlab.cnes.fr/OutilsCommuns/CorrelateurChaine3D/pandora).


```bash
    usage: pandora [-h] [-v] config

    Pandora stereo matching
    
    positional arguments:
      output_dir          Path to the output director
      config              Path to a json file containing the input files paths and the algorithm parameters
    
    optional arguments:
      -h, --help          show this help message and exit
      -v, --verbose       Increase output verbosity
```

Les fichiers de configuration pour les réseaux mc-cnn fast et accurate sont disponibles dans le dossier conf/. 
Il faut renseigner le chemin des poids des réseaux pré-entraînés.
Les poids des réseaux mc-cnn fast et accurate entrainés sur Middlebury sont disponibles dans le dossier weights/.
