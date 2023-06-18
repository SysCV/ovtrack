import h5py
import numpy as np
import os
from mmcv import BaseStorageBackend, FileClient


@FileClient.register_backend("hdf5", force=True)
class HDF5Backend(BaseStorageBackend):
    def __init__(self, img_db_path=None, vid_db_path=None, type="tao", **kwargs):

        # h5 file path
        self.img_db_path = img_db_path
        self.vid_db_path = vid_db_path

        self.img_client = None
        self.vid_client = None
        self.type = type

    def get(self, filepath):
        """Get values according to the filepath.
        Args:
            filepath (str | obj:`Path`): Here, filepath is the lmdb key.
        # """

        filepath = str(filepath)
        if self.type == "tao":
            if self.img_client is None and self.img_db_path is not None:
                self.img_client = h5py.File(self.img_db_path, "r")
            key_list = filepath.split("/")
            value_buf = np.array(
                self.img_client[key_list[-4]][key_list[-3]][key_list[-2]][key_list[-1]]
            )
        elif self.type == "key":
            if self.img_client is None and self.img_db_path is not None:
                self.img_client = h5py.File(self.img_db_path, "r")
            value_buf = self.img_client[filepath]
        elif self.type == "lvis":
            if self.img_client is None and self.img_db_path is not None:
                self.img_client = h5py.File(self.img_db_path, "r")
            filefolder, filename = os.path.split(filepath)
            value_buf = np.array(self.img_client[filename])
        elif self.type == "lasot":
            if self.img_client is None and self.img_db_path is not None:
                self.img_client = h5py.File(self.img_db_path, "r")
            key_list = filepath.split("/")
            value_buf = np.array(
                self.img_client[key_list[-4]][key_list[-3]][key_list[-2]][key_list[-1]][
                    "raw"
                ]
            )[0]
        elif self.type == "bdd":
            filefolder, filename = os.path.split(filepath)
            path, group_name = os.path.split(filefolder)

            if self.vid_client is None and self.vid_db_path is not None:
                self.vid_client = h5py.File(self.vid_db_path, "r")
            if self.img_client is None and self.img_db_path is not None:
                self.img_client = h5py.File(self.img_db_path, "r")
            if "/100k/" in filefolder:
                value_buf = np.array(self.img_client[filename])
            else:
                group = self.vid_client[group_name]
                value_buf = np.array(group[filename])

        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError
