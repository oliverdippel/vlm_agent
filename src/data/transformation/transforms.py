import torchvision.transforms as T


def get_train_transform(img_size: int = 224, augment: bool = False):
    transforms = [
        T.ToPILImage(),
        T.Resize((img_size, img_size), antialias=True),
    ]

    if augment:
        transforms.append(
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        )

    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    return T.Compose(transforms)