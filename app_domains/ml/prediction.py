"""Script that handles the crawler's ML predictions"""
from os.path import join
import pickle

from scipy.sparse import csr_matrix

from ml.feature_eng import Featurer


def load_model(config):
    assert "MAIN_DIR" in config
    assert "name_model" in config

    # ml
    with open(join(config["MAIN_DIR"], "input", config["name_model"] + ".pkl"), "rb+") as f:
        MDL = pickle.load(f)
    MDL.set_params(**{"n_jobs": 1})

    return MDL


class Predictor():
    """Object that handles the ML prediction pipeline"""
    def __init__(self, config, do_full_pipeline=False):

        MDL = load_model(config)

        self.MDL = MDL
        self.full_pipe_ready = False
        self.featurer = None

        if do_full_pipeline:
            featurer = Featurer(config)
            self.featurer = featurer
            self.full_pipe_ready = True

    def predict(self, x): # features in a CSR matrix
        """Predict parking classification of domain with features x"""
        x_csr = csr_matrix(x)
        pred_ml_park = self.MDL.predict(x_csr)[0]
        return pred_ml_park > 0.5

    def full_predict(self):
        if not self.full_pipe_ready:
            raise ValueError("Full pipe not initiated")
        pass
