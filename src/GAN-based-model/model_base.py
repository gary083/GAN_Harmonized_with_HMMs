from abc import ABC, abstractmethod

from lib.data_load import AttrDict, DataLoader


class ModelBase(ABC):

    @abstractmethod
    def train(
            self,
            args: AttrDict,
            data_loader: DataLoader,
            dev_data_loader: DataLoader = None,
            **kwargs,
    ):
        pass

    @abstractmethod
    def restore(self, save_dir):
        pass

    @abstractmethod
    def output_framewise_prob(self, output_path, data_loader):
        pass
