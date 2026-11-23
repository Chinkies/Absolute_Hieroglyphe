import albumentations as A
import cv2 as cv
from pathlib import Path

dataset_path = Path("data/Image_DAO_Sith")
out_dir = Path("dataset/augmented/")
images = []
bugged = 0


angle_limit = 10
rgb_limit = 30
N_VARIANTES = 10
transform = A.Compose(
    [
        A.Rotate(limit=(-angle_limit, angle_limit), p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.ToSepia(p=0.4),
        A.RGBShift(
            p=0.3,
            r_shift_limit=(-angle_limit, angle_limit),
            g_shift_limit=(-angle_limit, angle_limit),
            b_shift_limit=(-angle_limit, angle_limit),
        ),
    ]
)

if dataset_path.is_dir():
    for p in sorted(dataset_path.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            image = cv.imread(str(p))
            if image is None:
                bugged += 1
                print("l'image ", p.name, " est illisible")
                continue
            # print("img: ", p.name, ", infos: ", image.shape)
            images.append((image, p.stem))
        else:
            print("ceci n'est pas mon image: ", p.name)
print(len(images), "/", len(images) + bugged, " images correctement chargées")

# Augmentation des images
out_dir.mkdir(parents=True, exist_ok=True)
total = len(images)
for i, (image, name) in enumerate(images, start=1):
    for k in range(N_VARIANTES):
        img_aug = transform(image=image)["image"]
        cv.imwrite(str(out_dir / f"{name}_aug{k}.png"), img_aug)
        print(f"\rAugmentation : {i}/{total} ({100 * i // total}%)", end="\r", flush=True)
print("Augmentation terminée !")
