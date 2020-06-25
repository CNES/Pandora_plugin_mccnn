# Plugin_MC-CNN

Plugin MC-CNN : il permet d'intégrer le réseau mc-cnn dans Pandora.

## Installation

**Mode non-développeur**

Cette procédure vous permet d'installer les plugin_mc-cnn, pandora et mc-cnn, sans les cloner au préalable. 
Notez que les sources ne seront pas accessibles avec cette procédure.

Pour l'installer, suivez les étapes :

```sh
u@m $ python -m venv myEnv
u@m $ source myEnv/bin/activate
(myEnv) u@m $ pip install --upgrade pip
(myEnv) u@m $ pip install numpy
(myEnv) u@m $ pip install plugin_mc-cnn
```

**Mode développeur**

Cette procédure vous permet d'installer le plugin_mc-cnn, pandora, mc-cnn et d'avoir accès aux sources.

Pour l'installer, suivez les étapes :

- Initialiser l'environnement

```sh
u@m $ python -m venv myEnv
u@m $ source myEnv/bin/activate
(myEnv) u@m $ pip install --upgrade pip
(myEnv) u@m $ pip install numpy
```

- Installation de Pandora

```sh
(myEnv) u@m $ git clone https://github.com/CNES/Pandora_pandora.git
(myEnv) u@m $ cd Pandora_pandora
(myEnv) u@m $ pip install .
```

- Installation de mc-cnn 

```sh
(myEnv) u@m $ git clone https://gitlab.cnes.fr/OutilsCommuns/CorrelateurChaine3D/mc-cnn.git
(myEnv) u@m $ cd mc-cnn
(myEnv) u@m $ pip install .
```

- Installation du plugin

```sh
(myEnv) u@m $ git clone https://gitlab.cnes.fr/OutilsCommuns/CorrelateurChaine3D/pandora_plugins/plugin_mc-cnn.git
(myEnv) u@m $ cd plugin_mc-cnn
(myEnv) u@m $ pip install .
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
