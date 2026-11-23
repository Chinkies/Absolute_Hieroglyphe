import albumentations as A
import cv2 as cv
from pathlib import Path
import re


def augment_image(img, name, alb_compose, nb_variants=10):
    imgs = []
    for k in range(nb_variants):
        img_aug = alb_compose(image=img)["image"]
        imgs.append((img_aug, f"{name}_aug{k}.png"))
    return imgs


def reduce_folder(path):
    folder_path = Path(path)
    if not folder_path.is_dir():
        raise RuntimeError(f"Erreur: le dossier {path} n'existe pas !")
    pattern = re.compile(r"^(?P<base>.+)_aug(?P<k>\d+)\.png$")
    augmented = total = 0
    for p in sorted(folder_path.iterdir()):
        if not p.is_file():
            continue
        if pattern.match(p.name):
            p.unlink()
            augmented += 1
        total += 1
    print(f"Nettoyé {augmented} images sur {total} entrées")


def save_imgs(path, imgs):
    for img, name in imgs:
        cv.imwrite(str(path / name), cv.cvtColor(img, cv.COLOR_RGB2BGR))


def augment_folder(path_in, path_out, alb_compose, nb_variants=10):
    input_path = Path(path_in)
    output_path = Path(path_out)
    if not input_path.is_dir():
        raise RuntimeError(f"Erreur: le dossier {path_in} n'existe pas !")
    if not output_path.is_dir():
        raise RuntimeError(f"Erreur: le dossier {path_out} n'existe pas !")
    print("Augmenting folder", path_in, end="")
    if (path_in == path_out):
        print()
    else:
        print(" into", path_out)
    reduce_folder(path_out)
    pattern = re.compile(r"^(?P<base>.+)_aug(?P<k>\d+)\.png$")
    bugged = processed = total = 0
    for p in sorted(input_path.iterdir()):
        total += 1
        if p.suffix.lower() != ".png":
            print("\r\033[K", end="", flush=True)
            print("fichier ignoré : ", p.name)
        elif pattern.match(p.name):
            print("\r\033[K", end="", flush=True)
            print(
                f"Attention: données déjà augmentée détectées ! ({p.name})\nNettoyez le dataset"
            )
        else:
            image = cv.imread(str(p), )
            if image is None:
                bugged += 1
                print("\r\033[K", end="", flush=True)
                print("l'image ", p.name, " est inutilisable")
                continue
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            processed += 1
            augmented_images = augment_image(image, p.stem, alb_compose, nb_variants)
            save_imgs(output_path, augmented_images)
        print(
            f"\rAugmenting... ok: {processed}, ko: {bugged}, ign: {total - processed - bugged}, total: {total}",
            end="\r",
            flush=True,
        )
    print("\r\033[K", end="", flush=True)
    print(
        f"Augmentation done ! ok: {processed}, ko: {bugged}, ign: {total - processed - bugged}, total: {total}"
    )


dataset_path = "data/Image_DAO_Sith"
out_dir = "dataset/augmented/"

angle_limit = 10
rgb_limit = 30
transform = A.Compose(
    [
        A.Rotate(limit=(-angle_limit, angle_limit), p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.ToSepia(p=0.4),
        A.RGBShift(
            p=0.3,
            r_shift_limit=(-rgb_limit, rgb_limit),
            g_shift_limit=(-rgb_limit, rgb_limit),
            b_shift_limit=(-rgb_limit, rgb_limit),
        ),
    ]
)

# augment_folder(dataset_path, out_dir, transform)
# reduce_folder("dataset/yes")
augment_folder("dataset/yes/", "dataset/yes/", transform)
augment_folder("dataset/no/", "dataset/no/", transform)
