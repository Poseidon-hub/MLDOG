from torch.utils.data import Dataset
from torchvision import datasets


SELECTED_BREEDS = [
    "Beagle",
    "Pug",
    "Samoyed",
    "Shiba Inu",
    "Yorkshire Terrier",
]


class FiveBreeds(Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

        name_to_old = {name: i for i, name in enumerate(base.classes)}
        selected_old = [name_to_old[name] for name in SELECTED_BREEDS]
        self.old_to_new = {old: new for new, old in enumerate(selected_old)}
        self.indices = [
            i for i, (_, label) in enumerate(base)
            if label in self.old_to_new
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, old_label = self.base[self.indices[index]]
        return self.transform(image), self.old_to_new[old_label]


def load_datasets(root, transform):
    train_all = datasets.OxfordIIITPet(
        root=root,
        split="trainval",
        target_types="category",
        download=True,
    )
    test_all = datasets.OxfordIIITPet(
        root=root,
        split="test",
        target_types="category",
        download=True,
    )
    print(len(train_all), len(test_all))
    print(train_all.classes)
    return FiveBreeds(train_all, transform), FiveBreeds(test_all, transform)
