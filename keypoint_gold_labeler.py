import configparser
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

from lib import common

game_config_path = '/home/ebensh/src/pinball_cv/configs/hot_hand_full_resolution.cfg'  # For running in Atom
path_base = 'hot_hand_ebensh_2017_11_27'
game_config = configparser.ConfigParser(defaults={path_base: path_base})
game_config.read(game_config_path)
input_rows = game_config.getint('PinballFieldVideo', 'rows')
input_cols = game_config.getint('PinballFieldVideo', 'cols')
