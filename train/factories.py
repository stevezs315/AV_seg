from utils_pytorch import UniversalFactory
import models
import losses


class ModelFactory(UniversalFactory):
    classes = [
        models.RRWNet,
    ]


class LossesFactory(UniversalFactory):
    classes = [
        losses.BCE3Loss,
        losses.BCE3wminmaxLoss,
        losses.RRLoss,
        losses.SpCLLoss,
    ]
