import os

__TEACHERS_DIR = os.path.join(os.path.dirname(__file__), 'teachers')


def get_checkpoint_filename(model_name: str, dataset_name: str):
    model_name = model_name.strip().lower()
    dataset_name = dataset_name.strip().lower()
    return os.path.join(__TEACHERS_DIR, model_name), dataset_name + '.pt'

from models import classifier
