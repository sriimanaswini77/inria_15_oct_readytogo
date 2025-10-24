import cv2
import numpy as np
from seaborn import color_palette
import torch
import torch.nn.functional as F
from torchvision import models, transforms, utils
import copy
from utils import *

def slope(p1, p2):
    if p2[0] == p1[0]:
        return float('inf')
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def filter_by_slope(points, expected_slope, tolerance=0.1):
    filtered_points = []
    for i in range(1, len(points)):
        s = slope(points[i-1], points[i])
        if abs(s - expected_slope) <= tolerance:
            filtered_points.append(points[i])
    return filtered_points

def get_next_point(score_map, prev_point, threshold=50):
    flat_scores = score_map.flatten()
    top_indices = np.argpartition(flat_scores, -3)[-3:]
    top_indices = top_indices[np.argsort(-flat_scores[top_indices])]
    top_points = [np.unravel_index(idx, score_map.shape) for idx in top_indices]

    prev_x, prev_y = prev_point
    distances = []
    for (y, x) in top_points:
        dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        distances.append(dist)

    nearest_index = np.argmin(distances)
    nearest_point = top_points[nearest_index]
    nearest_distance = distances[nearest_index]

    if nearest_distance <= threshold:
        return (nearest_point[1], nearest_point[0]), True
    else:
        return (nearest_point[1], nearest_point[0]), False

class ImageData():
    # Your existing ImageData code here, unchanged
    pass

class VideoDataset2():
    # Your existing VideoDataset2 code here, unchanged
    pass

class Featex():
    # Your existing Featex code here, unchanged
    pass

class CreateModel():
    def __init__(self, alpha, model, use_cuda):
        self.alpha = alpha
        self.featex = Featex(model, use_cuda)
        self.I_feat = None

    def __call__(self, template, image):
        T_feat = self.featex(template)
        self.I_feat = self.featex(image)
        conf_maps = None
        batchsize_T = T_feat.size()[0]

        for i in range(batchsize_T):
            T_feat_i = T_feat[i].unsqueeze(0)
            I_feat_norm, T_feat_i = MyNormLayer(self.I_feat, T_feat_i)
            dist = torch.einsum(
                "xcab,xcde->xabde",
                I_feat_norm / torch.norm(I_feat_norm, dim=1, keepdim=True),
                T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True)
            )
            conf_map = QATM(self.alpha)(dist)
            if conf_maps is None:
                conf_maps = conf_map
            else:
                conf_maps = torch.cat([conf_maps, conf_map], dim=0)
        return conf_maps

class QATM():
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        batch_size, ref_row, ref_col, qry_row, qry_col = x.size()
        x = x.view(batch_size, ref_row * ref_col, qry_row * qry_col)
        xm_ref = x - torch.max(x, dim=1, keepdim=True)[0]
        xm_qry = x - torch.max(x, dim=2, keepdim=True)[0]
        confidence = torch.sqrt(
            F.softmax(self.alpha * xm_ref, dim=1) *
            F.softmax(self.alpha * xm_qry, dim=2)
        )
        conf_values, ind3 = torch.topk(confidence, 1)
        ind1, ind2 = torch.meshgrid(torch.arange(batch_size), torch.arange(ref_row * ref_col))
        ind1, ind2, ind3 = ind1.flatten(), ind2.flatten(), ind3.flatten()
        values = confidence[ind1, ind2, ind3]
        values = torch.reshape(values, [batch_size, ref_row, ref_col, 1])
        return values

def run_one_sample(model, template, image, prev_point=None):
    val = model(template, image)
    if val.is_cuda:
        val = val.cpu()
    val = val.numpy()
    val = np.log(val)
    batch_size = val.shape[0]
    scores = []

    for i in range(batch_size):
        gray = val[i, :, :, 0]
        gray = cv2.resize(gray, (image.size()[-1], image.size()[-2]))
        h = template.size()[-2]
        w = template.size()[-1]
        score = np.exp(gray / (h * w))
        scores.append(score)

        if prev_point is not None:
            next_point, is_valid = get_next_point(score, prev_point, threshold=50)
            if not is_valid:
                print("Deviation beyond threshold — using nearest alternative:", next_point)
            else:
                print("Next point consistent:", next_point)
            prev_point = next_point

    return np.array(scores), prev_point

# Rest of your unchanged functions (nms, nms_multi, plot_result, etc.) would go here

# Model initialization
model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=25, use_cuda=True)

# Example usage (to be incorporated as needed):
# filtered_scores, last_point = run_one_sample(model, template, image, prev_point=(x0, y0))

