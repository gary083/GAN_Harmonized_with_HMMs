from .uns import UnsModel
from .sup import SupModel
from .uns_bert_ import UnsBertModel


MODEL_HUB = {
    'uns': UnsModel,
    'sup': SupModel,
    'uns_bert': UnsBertModel,
}
