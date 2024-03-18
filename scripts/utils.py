import argparse
import numpy as np
import torch
from distutils.util import strtobool
from torchmetrics import Metric
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

torch.set_printoptions(precision=3, sci_mode=False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=int, default=0, help='database index')
    parser.add_argument('--n_clusters', type=int, default=15, help='cluster num for k-means')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size used for training and testing')
    parser.add_argument('--threshold', type=float, default=-1.0, help='Threshold used for training and testing')
    parser.add_argument('--is_remove', type=strtobool, default=True)
    parser.add_argument('--is_standarize', type=strtobool, default=True)
    parser.add_argument('--is_filter', type=strtobool, default=True)
    parser.add_argument('--crop', type=int, default=0)
    parser.add_argument('--min_count', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--no_log', action='store_true', help='Do not log')
    args = parser.parse_args()
    
    return args

def create_cityscapes_colormap(num_classes=27):
    if num_classes == 19:
        colors = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142), (0, 0, 0)]
    else:
        colors = [(128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153), (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142), (0, 0, 0)]
    return np.array(colors, dtype=int)[:,::-1]

def create_semantic_segmentation_map(num_classes=128):
    segmentation_map = np.random.randint(0, 256, (num_classes, 3), dtype=np.uint8)
    return segmentation_map

class UnsupervisedMetrics(Metric):
    def __init__(
        self,
        prefix: str,
        n_classes: int,
        extra_clusters: int,
        compute_hungarian: bool,
        dist_sync_on_step=True,
    ):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_classes = n_classes
        self.extra_clusters = extra_clusters
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        self.add_state(
            "stats",
            default=torch.zeros(
                n_classes + self.extra_clusters, n_classes, dtype=torch.int64
            ),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1)
            preds = preds.reshape(-1)
            mask = (
                (actual >= 0)
                & (actual < self.n_classes)
                & (preds >= 0)
                & (preds < self.n_classes)
            )
            actual = actual[mask]
            preds = preds[mask]
            self.stats += (
                torch.bincount(
                    (self.n_classes + self.extra_clusters) * actual + preds,
                    minlength=self.n_classes * (self.n_classes + self.extra_clusters),
                )
                .reshape(self.n_classes, self.n_classes + self.extra_clusters)
                .t()
                .to(self.stats.device)
            )

    def map_clusters(self, clusters):
        if self.extra_clusters == 0:
            return torch.tensor(self.assignments[1])[clusters]
        else:
            missing = sorted(
                list(
                    set(range(self.n_classes + self.extra_clusters))
                    - set(self.assignments[0])
                )
            )
            cluster_to_class = self.assignments[1]
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, -1)
                else:
                    cluster_to_class = np.insert(
                        cluster_to_class, missing_entry + 1, -1
                    )
            cluster_to_class = torch.tensor(cluster_to_class)
            return cluster_to_class[clusters]

    def compute(self):
        if self.compute_hungarian:
            self.assignments = linear_sum_assignment(
                self.stats.detach().cpu(), maximize=True
            )
            # print(self.assignments)
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]
            if self.extra_clusters > 0:
                self.assignments_t = linear_sum_assignment(
                    self.stats.detach().cpu().t(), maximize=True
                )
                histogram = self.stats[self.assignments_t[1], :]
                missing = list(
                    set(range(self.n_classes + self.extra_clusters))
                    - set(self.assignments[0])
                )
                new_row = self.stats[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], axis=0)
                new_col = torch.zeros(self.n_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], axis=1)
        else:
            self.assignments = (
                torch.arange(self.n_classes).unsqueeze(1),
                torch.arange(self.n_classes).unsqueeze(1),
            )
            self.histogram = self.stats

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)

        metric_dict = {
            self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
            self.prefix + "Accuracy": opc.item(),
            "IoU": iou,
        }
        return {k: 100 * v for k, v in metric_dict.items()}


class ClassAssignment:
    def __init__(self, gt_class_name):
        self.sbert = SentenceTransformer('all-mpnet-base-v2').cuda()
        self.sbert.eval()
        # torch.Size([20, 768])
        self.gt_embeddings = self.sbert.encode(gt_class_name, convert_to_tensor=True)

    def __call__(self, logits, class_names, threshold=None):
        # logits, torch.Size([b, n, 448, 448])
        segments = []
        for class_name, segment in zip(class_names, logits.argmax(dim=1)):
            embedding = self.sbert.encode(class_name, convert_to_tensor=True)
            nc = torch.einsum("nf,cf->nc", embedding, self.gt_embeddings)
            class_assign = nc.argmax(dim=-1)
            if threshold is not None:
                class_assign[nc.max(dim=-1).values < threshold] = -1
            segments.append(class_assign[segment][None, ...])
        return torch.cat(segments, dim=0)
 

class CountClassPerImage:
    def __init__(self):
        self.count = []
    def regist(self, target):
        # b, h, w
        for t in target:
            tu = list(map(int, list(t.unique())))
            if -1 in tu:
                tu.remove(-1)
            self.count.append(len(tu))
    def compute(self):
        return sum(self.count) / len(self.count)
