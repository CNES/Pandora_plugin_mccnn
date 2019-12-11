# coding: utf-8
"""
:author: Véronique Defonte
:organization: CS SI
:copyright: 2019 CNES. All rights reserved.
:created: dec. 2019
"""
import logging

from pandora.stereo import stereo
from mc_cnn.run import run_mc_cnn_fast, run_mc_cnn_accurate

@stereo.AbstractStereo.register_subclass('mc_cnn')
class MCCNN(stereo.AbstractStereo):
    """

    MCCNN class is a plugin that create a cost volume by calling the McCNN library: a neural network that produce a
    similarity score

    """
    _architecture = 'fast'
    _window_size = 11
    _subpix = 1
    _trained_net = None

    def __init__(self, **cfg):
        """

        :param cfg: optional configuration, {'stereo_method': value, 'mc_cnn_arch': 'fast' | 'accurate' }
        :type cfg: dictionary
        """
        self.check_config(**cfg)

    def check_config(self, **cfg):
        """
        Check and update the configuration

        :param cfg: optional configuration, {'mc_cnn_arch': 'fast' | 'accurate' }
        :type cfg: dictionary
        """
        if 'mc_cnn_arch' in cfg:
            if cfg['mc_cnn_arch'] == 'fast':
                self._architecture = 'fast'
            elif cfg['mc_cnn_arch'] == 'accurate':
                self._architecture = 'accurate'
            else:
                logging.error("No mc-cnn architecture name {} supported".format(cfg['mc_cnn_arch']))
                exit()

        if 'window_size' in cfg:
            if int(cfg['window_size']) != 11:
                logging.error("Mc-cnn similarity measure only accepts window_size = 11")
                exit()

        if 'subpix' in cfg:
            if int(cfg['subpix']) != 1:
                logging.error("Mc-cnn similarity measure only accepts subpixel = 1")
                exit()

        if not 'model_path' in cfg:
            logging.error("A path to a trained network is required")
            exit()
        else:
            self._trained_net = cfg['model_path']

    def desc(self):
        """
        Describes the optimization method

        """
        print('MC-CNN similarity measure')

    def compute_cost_volume(self, img_ref, img_sec, disp_min, disp_max, **cfg):
        """
        Computes the cost volume for a pair of images

        :param img_ref: reference Dataset image
        :type img_ref:
        xarray.Dataset containing :
            - im : 2D (row, col) xarray.DataArray
        :param img_sec: secondary Dataset image
        :type img_sec:
        xarray.Dataset containing :
            - im : 2D (row, col) xarray.DataArray
        :param disp_min: minimum disparity
        :type disp_min: int
        :param disp_max: maximum disparity
        :type disp_max: int
        :param cfg: images configuration containing the mask convention : valid_pixels, no_data
        :type cfg: dict
        :return: the cost volume
        :rtype: xarray.Dataset, with the data variables cost_volume 3D xarray.DataArray (row, col, disp)
        """
        if self._architecture == 'fast':
            cv = run_mc_cnn_fast(img_ref, img_sec, disp_min, disp_max, self._trained_net)

            # TODO : supprimer ces lignes quand on aura accès à la libsgm en float
            # Because sgm takes a cost volume of type uint8
            # Before : cost range into (-1, 1)
            # After : cost range into (0, 2)
            #cv += 1
            # After : cost range into (0, 200)
            #cv *= 100
        # Accurate architecture
        else:
            cv = run_mc_cnn_accurate(img_ref, img_sec, disp_min, disp_max, self._trained_net)

        # Create the xarray.DataSet that will contain the cost_volume of dimensions (row, col, disp)
        metadata = {"measure": 'mc_cnn_' + self._architecture, "subpixel": self._subpix,
                    "offset_row_col": int((self._window_size - 1) / 2), "window_size": self._window_size}

        # Allocate the xarray cost volume
        cv = self.allocate_costvolume(img_ref, self._subpix, disp_min, disp_max, self._window_size, metadata, cv)
        return cv
