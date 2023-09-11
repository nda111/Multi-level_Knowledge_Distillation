import os
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, CocoDetection

__DATA_DIR = os.path.dirname(__file__)
__IMAGENET_ROOT = '/material/data/imagenet-original'
__IMAGENET_CLASSES_FILENAME = os.path.join(__DATA_DIR, 'imagenet1k_classes.txt')
__COCODET_ROOT = None

def get_mean_std(dataset_name: str):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif dataset_name == 'cifar100':
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif dataset_name == 'imagenet':
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif dataset_name == 'cocodet':
        raise NotImplementedError
    else:
        raise NameError(f"'{dataset_name}' is not supported.")
    
    
def get_classes(dataset_name: str):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        return get_dataset('cifar10').classes
    elif dataset_name == 'cifar100':
        return get_dataset('cifar100').classes
    elif dataset_name == 'imagenet':
        with open(__IMAGENET_CLASSES_FILENAME, 'r') as file:
            return eval(file.readline())
    elif dataset_name == 'cocodet':
        raise NotImplementedError
    else:
        raise NameError(f"'{dataset_name}' is not supported.")


def get_dataset(dataset_name: str, train: bool=True, transform=None, target_transform=None, download: bool=True):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        return CIFAR10(__DATA_DIR, train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'cifar100':
        return CIFAR100(__DATA_DIR, train=train, transform=transform, target_transform=target_transform, download=download)
    elif dataset_name == 'imagenet':
        root = os.path.join(__IMAGENET_ROOT, 'train' if train else 'val')
        return ImageFolder(root, transform=transform, target_transform=target_transform)
    elif dataset_name == 'cocodet':
        raise NotImplementedError
        # return CocoDetection(__DATA_DIR, annFile=)
    else:
        raise NameError(f"'{dataset_name}' is not supported.")
