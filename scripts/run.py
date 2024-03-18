import datetime
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from tag_utils import create_tag_model, available_models
from scripts.dino_kmeans import save_logits
from scripts.load_dataset import load_voc_dataloader
from scripts.utils import get_args, create_semantic_segmentation_map, UnsupervisedMetrics, ClassAssignment, CountClassPerImage
from retrieve_utils.src.data.components.transforms import default_vocab_transform


args = get_args()
no_log = args.no_log
epoch_num = 1
batch_size = args.batch_size
image_size = 448
class_num = 20
crop = args.crop
use_crf = True
n_clusters = args.n_clusters
num_samples = args.num_samples
threshold = args.threshold
database_name = ["ViT-L-14_PMD_TOP5", "ViT-L-14_CC12M", "ViT-L-14_ENGLISH_WORDS", "ViT-L-14_WORDNET"][args.database]
vocab_transform = default_vocab_transform(is_remove=args.is_remove, is_standardize=args.is_standarize, is_filter=args.is_filter, min_count=args.min_count)

print(f"VOC: {class_num} classes, {n_clusters} clusters, {database_name}, crf {use_crf}, batch size {batch_size}, image size {image_size}, crop {crop}, num_samples {num_samples}")
print(args.is_remove, args.is_standarize, args.is_filter, args.min_count)

now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d-%H-%M")
dir_for_output = "outputs/" + current_time
os.makedirs(dir_for_output, exist_ok=True)
print(dir_for_output)

if not no_log:
    print(available_models())

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "ViT-L/14"
pretrained = 'openai'
tag_model = create_tag_model(model_name=model_name, pretrained=pretrained, device=device, database_name=database_name, n_clusters=n_clusters, vocab_transform=vocab_transform, num_samples=num_samples)
tag_model.eval()

val_dataloader = load_voc_dataloader(is_val=True, batch_size=batch_size, image_size=image_size, crop=crop)
gt_class_name = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']

ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
semseg_metrics = UnsupervisedMetrics("origin/semseg/", class_num, 0, False)
ca = ClassAssignment(gt_class_name)
cc = CountClassPerImage()

colors = create_semantic_segmentation_map(500)

for idx, (images, targets) in tqdm(enumerate(val_dataloader, 0)):
    images = images.cuda()
    targets = targets.cuda()
    cc.regist(targets)

    with torch.no_grad():
        seg_logits, class_names, dino_logits = tag_model.batched_forward_cased(images, use_crf=use_crf)
        semseg_metrics.update(ca(dino_logits, class_names, threshold), targets)
    
    if not no_log:
        print(cc.compute())
        print(idx, semseg_metrics.compute())
    save_logits(dino_logits, class_names, dir_for_output, idx * batch_size)
    for i, class_name in enumerate(class_names):
        image_id = idx * batch_size + i
        dino_images = dino_logits[i].argmax(dim=0)
        p = dino_images.cpu().detach().numpy()
        cv2.imwrite(f"{dir_for_output}/dino_kmeans_{image_id}.png", colors[p])
        rgb_img = np.array(torch.permute(images[i], (1, 2, 0)).cpu())[:,:,::-1]
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min()) * 255
        cv2.imwrite(f"{dir_for_output}/val_input_{image_id}.png", rgb_img)
        pred = targets[i].cpu().detach().numpy()
        cv2.imwrite(f"{dir_for_output}/val_targets_{image_id}.png", colors[pred])

print(cc.compute())
print(semseg_metrics.compute())





