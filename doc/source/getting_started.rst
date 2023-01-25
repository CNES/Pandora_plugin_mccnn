Getting started
===============

Overview
########

`Pandora <https://github.com/CNES/Pandora>`_ stereo matching framework is designed to provide some state of the art stereo algorithms and to add others one as plugins.
This `Pandora plugin <https://pandora.readthedocs.io/en/stable/userguide/plugin.html>`_ aims to compute the cost volume using the similarity measure produced by MC-CNN neural network, defined by [Zbontar]_, with the `MCCNN <https://github.com/CNES/Pandora_MCCNN>`_ library .

.. [Zbontar] Zbontar, J., & LeCun, Y. (2016). Stereo matching by training a convolutional neural network to compare image patches. J. Mach. Learn. Res., 17(1), 2287-2318.

Install
#######

**pandora_plugin_mccnn** is available on Pypi and can be installed by:

.. code-block:: bash

    pip install pandora_plugin_mccnn


This command will installed required dependencies as `Pandora <https://github.com/CNES/Pandora>`_ and `MCCNN <https://github.com/CNES/Pandora_MCCNN>`_.

Usage
#####


Let's refer to `Pandora's readme <https://github.com/CNES/Pandora/blob/master/README.md>`_ or `online documentation <https://pandora.readthedocs.io/?badge=latest>`_ for further information about Pandora general functionalities.

More specifically, you can find :

- `MCCNN configuration file example <https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/json_conf_files/a_semi_global_matching_with_mccnn_similarity_measure.json>`_

- `documentation about MCCNN theory and parameters <https://pandora.readthedocs.io/en/stable/userguide/plugins/plugin_mccnn.html>`_


Pretrained Weights for MCCNN networks
#####################################

Pretrained weights for mc-cnn fast and mc-cnn accurate neural networks are available in the weights directory :

-  mc_cnn_fast_mb_weights.pt and mc_cnn_accurate_mb_weights.pt are the weights of the pretrained networks on the Middlebury dataset [Middlebury]_

-  mc_cnn_fast_data_fusion_contest.pt and mc_cnn_accurate_data_fusion_contest.pt are the weights of the pretrained networks on the Data Fusion Contest dataset [DFC]_

.. [Middlebury] Scharstein, D., Hirschmüller, H., Kitajima, Y., Krathwohl, G., Nešić, N., Wang, X., & Westling, P. (2014, September). High-resolution stereo datasets with subpixel-accurate ground truth. In German conference on pattern recognition (pp. 31-42). Springer, Cham.

.. [DFC] Bosch, M., Foster, K., Christie, G., Wang, S., Hager, G. D., & Brown, M. (2019, January). Semantic stereo for incidental satellite images. In 2019 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 1524-1532). IEEE.

Related
#######

* `Pandora <https://github.com/CNES/Pandora>`_ - A stereo matching framework

* `MCCNN <https://github.com/CNES/Pandora_MCCNN>`_ - Pytorch/python implementation of mc-cnn neural network


References
##########

Please cite the following paper when using Pandora and pandora_plugin_mccnn:

*Cournet, M., Sarrazin, E., Dumas, L., Michel, J., Guinet, J., Youssefi, D., Defonte, V., Fardet, Q., 2020. Ground-truth generation and disparity estimation for optical satellite imagery. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences.*
