# import os
# import random
# import shutil


# def create_validation_split(data_dir: str, split_ratio: float = 0.2, seed: int = 42):
#     random.seed(seed)

#     train_dir = os.path.join(data_dir, "train")
#     val_dir = os.path.join(data_dir, "val")

#     for cls in ["real", "fake"]:
#         train_cls_dir = os.path.join(train_dir, cls)
#         val_cls_dir = os.path.join(val_dir, cls)

#         os.makedirs(val_cls_dir, exist_ok=True)

#         images = [
#             f for f in os.listdir(train_cls_dir)
#             if os.path.isfile(os.path.join(train_cls_dir, f))
#         ]

#         random.shuffle(images)
#         n_val = int(len(images) * split_ratio)
#         val_images = images[:n_val]

#         for image_name in val_images:
#             src = os.path.join(train_cls_dir, image_name)
#             dst = os.path.join(val_cls_dir, image_name)
#             shutil.move(src, dst)

#         print(f"{cls}: moved {n_val} images to validation")