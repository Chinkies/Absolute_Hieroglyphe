import albumentations as A
import cv2 as cv
from math import ceil
from statistics import median_high
from pathlib import Path
import re


def parse_folder(path):
    folder_path = Path(path)
    if not folder_path.is_dir():
        raise RuntimeError(f"Erreur: le dossier {path} n'existe pas !")
    return folder_path


def augment_image(img, name, alb_compose, nb_variants=10):
    imgs = []
    for k in range(nb_variants):
        img_aug = alb_compose(image=img)["image"]
        imgs.append((img_aug, f"{name}_aug{k}.png"))
    return imgs


def reduce_folder(path, recursive=False):
    folder_path = parse_folder(path)
    pattern = re.compile(r"^(?P<base>.+)_aug(?P<k>\d+)\.png$")
    augmented = total = 0
    for p in sorted(folder_path.iterdir()):
        if not p.is_file():
            if recursive and p.is_dir():
                reduce_folder(p)
            continue
        if pattern.match(p.name):
            p.unlink()
            augmented += 1
        total += 1
    print(f"Nettoyé {augmented} images sur {total} entrées")


def save_imgs(path, imgs):
    for img, name in imgs:
        cv.imwrite(str(path / name), cv.cvtColor(img, cv.COLOR_RGB2BGR))


AUG_PATTERN = re.compile(r"^(?P<base>.+)_aug(?P<k>\d+)\.png$")


def count_classes(root):
    counts = {}
    for class_dir in sorted(root.iterdir()):
        if class_dir.is_dir():
            counts[class_dir.name] = sum(
                1
                for f in class_dir.iterdir()
                if f.is_file() and not AUG_PATTERN.match(f.name)
            )
    return counts


def variants_needed(counts, base_variants):
    target = median_high(counts.values())
    goal = target * (base_variants)
    return {c: int(ceil(max(0, goal / n))) for c, n in counts.items()}


def augment_classes_even(class_root, alb_compose, base_variants=10):
    root = parse_folder(class_root)
    reduce_folder(root, recursive=True)
    count = count_classes(root)
    imagesToAdd = variants_needed(count, base_variants)
    print(f"Augmenting {len(count)} classes by {base_variants} (uncompensed)...")
    for folder, correction in imagesToAdd.items():
        class_path = root / folder
        print(
            f"Augmenting class {folder} by {correction} images per img"
        )
        augment_folder(class_path, class_path, alb_compose, correction)


def augment_folder(path_in, path_out, alb_compose, nb_variants=10):
    input_path = parse_folder(path_in)
    output_path = parse_folder(path_out)
    print("Augmenting folder", path_in, end="")
    if path_in == path_out:
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
            image = cv.imread(
                str(p),
            )
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


# Test transformer
angle_limit = 10
rgb_limit = 30
transform = A.Compose(
    [
        A.Rotate(limit=(-angle_limit, angle_limit), p=0.6),
        A.GaussNoise(std_range=(0.1, 0.2), p=0.5),
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

# dataset = parse_folder("dataset/")
# classCount = count_classes(dataset)

# imagesToAdd = variants_needed(classCount, 10)

# print(imagesToAdd)

augment_classes_even("dataset/", transform, 10)
