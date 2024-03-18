import cv2
import torch
import torch.nn.functional as F
import numpy as np


model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda()


def dino_kmeans(image, k_means):
    # torch.Size([1, 3, 448, 448])
    B, _, H, W = image.shape
    h_up, w_up = H, W
    # h_up, w_up = H*2, W*2
    # image = F.interpolate(image, size=(h_up, w_up), mode="bilinear", align_corners=False)
    feat = model.forward_features(image)['x_norm_patchtokens']
    kmeans_result = k_means(feat)
    # kmeans_result.labels, torch.Size([8, 257])
    # kmeans_result.centers, torch.Size([8, 10, 768])
    kmeans_centers = kmeans_result.centers
    
    image_feat = feat.reshape(B, h_up//14, w_up//14, -1)
    image_feat = torch.permute(image_feat, (0, 3, 1, 2))
    image_feat = F.interpolate(image_feat, size=(H, W), mode="bilinear", align_corners=False)
    
    logits = torch.einsum("bnf,bfhw->bnhw", kmeans_centers, image_feat)
    return logits

def extract_clip_features(clip_features, dino_logits):
    # clip_features, torch.Size([8, 1025, 768])
    # dino_logits, torch.Size([8, 15, 448, 448])
    B, _, feat_dim = clip_features.shape
    B, C, H, W = dino_logits.shape
    clip_features = clip_features[:, 1:, :].reshape(B, H//14, W//14, feat_dim)
    clip_features = torch.permute(clip_features, (0, 3, 1, 2))
    clip_features = F.interpolate(clip_features, size=(H, W), mode="bilinear", align_corners=False)
    
    class2feat = []
    dino_logits = F.one_hot(dino_logits.argmax(dim=1), num_classes=C).permute(0, 3, 1, 2)
    for i in range(C):
        class2feat.append((clip_features * dino_logits[:,i,:,:].unsqueeze(1) / dino_logits[:,i,:,:].unsqueeze(1).sum(dim=-1).sum(dim=-1).sum(dim=1)[:, None, None, None]).sum(dim=-1).sum(dim=-1))
    class2feats = torch.stack(class2feat, dim=1).cuda()
    # class2feats, torch.Size([8, 15, 768])
    return class2feats, clip_features

def get_clip_logits(class2feats, clip_features):
    # class2feats, torch.Size([8, 15, 768])
    # clip_features, torch.Size([8, 768, 448, 448])
    logits = torch.einsum("bnf,bfhw->bnhw", class2feats, clip_features)
    return logits

def drow_white_line(image):
    vertical_boundary = image[1:, :, :] != image[:-1, :, :]
    horizontal_boundary = image[:, 1:, :] != image[:, :-1, :]
    image[1:, :, :][vertical_boundary] = 255
    image[:, 1:, :][horizontal_boundary] = 255
    return image
    
    

def save_logits(logits, class_names, dir_for_output=None, image_id=None, font_thickness=2, font_scale=1.5):
    # logits, torch.Size([8, 15, 448, 448])
    # class_names, [8, 15]
    B, _, h, w = logits.shape
    segmentation_results = logits.argmax(dim=1).cpu().detach().numpy()
    for batch in range(B):
        class_mask = {}
        class_name = class_names[batch]
        result_image = np.zeros((h, w, 3), dtype=np.uint8)
        color_map = {name: np.random.randint(0, 255, (3,), dtype=np.uint8) for name in class_name}
        for class_index, name in enumerate(class_name):
            mask = segmentation_results[batch] == class_index
            color = color_map[name]
            result_image[mask] = color
            ys, xs = np.where(mask)
            center_x, center_y = xs.mean().astype(int), ys.mean().astype(int)
            if name in class_mask:
                class_mask[name] += mask
            else:
                class_mask[name] = mask
        
        drow_white_line(result_image)
        
        for name in set(class_name):
            ys, xs = np.where(class_mask[name])
            center_x, center_y = xs.mean().astype(int), ys.mean().astype(int)
            # cv2.rectangle(result_image, (max(0, center_x-35), max(0, center_y-10)), (min(h-1, center_x+35), min(w-1, center_y+10)), color_map[name], -1)
            # cv2.putText(result_image, name, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = font_scale
            font_thickness = font_thickness

            (text_width, text_height), baseline = cv2.getTextSize(name, font_face, font_scale, font_thickness)

            start_point = (int(center_x), int(center_y - text_height // 2 - baseline))
            end_point = (int(center_x + text_width), int(center_y + text_height // 2))
            
            try:
                cv2.rectangle(result_image, pt1=start_point, pt2=end_point, color=(0, 0, 0), thickness=cv2.FILLED)
                cv2.putText(result_image, name, (center_x, center_y), font_face, font_scale, tuple(color_map[name].tolist()), font_thickness)
            except:
                continue
        if (dir_for_output is not None) and (image_id is not None):
            cv2.imwrite(f"{dir_for_output}/val_semseg_{image_id + batch}.png", result_image)
        else:
            return result_image
