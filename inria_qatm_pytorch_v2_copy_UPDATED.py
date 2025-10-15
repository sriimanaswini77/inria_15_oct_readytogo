import cv2
import numpy as np
from seaborn import color_palette
import torch
import torch.nn.functional as F
from torchvision import models, transforms, utils
import copy
from utils import *

class ImageData():
    """
    Modified ImageData class with INS Point support for dynamic search region updates

    Workflow:
    1. Loads global satellite image
    2. Defines INS point manually near template location
    3. Creates cropped search region based on INS point
    4. Passes cropped region (not full image) to CreateModel
    5. Supports dynamic INS point updates for sequential propagation
    """
    def __init__(self, source_img, thres=0.7, transform=None, half=False, 
                 ins_point=None, search_region_size=(500, 500)):
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        self.half = half
        self.image_name = source_img

        # Load the GLOBAL image (full satellite image)
        self.image_raw_global = cv2.imread(self.image_name)
        print("Global image resolution:", self.image_raw_global.shape)

        # INS point for defining search region
        self.ins_point = ins_point  # Format: (x, y) coordinates
        self.search_region_size = search_region_size  # (width, height)

        # Store the cropped search region
        self.image_raw = None
        self.image = None

        # If INS point is provided, create cropped search region
        if self.ins_point is not None:
            self._create_search_region()
        else:
            # Use full global image if no INS point defined
            self.image_raw = self.image_raw_global.copy()
            self.image = self.transform(self.image_raw).unsqueeze(0)

        self.thresh = thres

        if self.half:
            self.image = self.image.half()

    def _create_search_region(self):
        """
        Create a cropped search region based on INS point
        INS point acts as the center/reference for the search area
        """
        ins_x, ins_y = self.ins_point
        region_w, region_h = self.search_region_size

        # Calculate crop boundaries centered around INS point
        x1 = max(0, ins_x - region_w // 2)
        y1 = max(0, ins_y - region_h // 2)
        x2 = min(self.image_raw_global.shape[1], ins_x + region_w // 2)
        y2 = min(self.image_raw_global.shape[0], ins_y + region_h // 2)

        # Store crop offset for coordinate mapping later
        self.crop_offset = (x1, y1)

        # Crop the search region from global image
        self.image_raw = self.image_raw_global[y1:y2, x1:x2].copy()
        print(f"Search region created at INS point ({ins_x}, {ins_y})")
        print(f"Cropped region size: {self.image_raw.shape}")
        print(f"Crop boundaries: x[{x1}:{x2}], y[{y1}:{y2}]")

        # Transform the cropped region
        self.image = self.transform(self.image_raw).unsqueeze(0)

    def set_ins_point(self, ins_point, search_region_size=None):
        """
        Manually define/update INS point and regenerate search region
        This enables DYNAMIC sequential search region updates

        Args:
            ins_point: (x, y) tuple - coordinates on global image
            search_region_size: optional (width, height) tuple
        """
        self.ins_point = ins_point
        if search_region_size is not None:
            self.search_region_size = search_region_size

        self._create_search_region()

        if self.half:
            self.image = self.image.half()

    def load_template(self, template):
        """
        Load template for matching
        Now this template will be matched against the CROPPED search region
        """
        if self.transform:
            template = self.transform(template)

        if self.half:
            template = template.half()

        return {
            'image': self.image,  # This is now the CROPPED region
            'image_raw': self.image_raw,  # Cropped region raw
            'image_raw_global': self.image_raw_global,  # Full global image
            'image_name': self.image_name,
            'template': template.unsqueeze(0),
            'template_h': template.size()[-2],
            'template_w': template.size()[-1],
            'thresh': self.thresh,
            'ins_point': self.ins_point,
            'crop_offset': self.crop_offset if hasattr(self, 'crop_offset') else (0, 0)
        }

    def map_to_global_coords(self, local_x, local_y):
        """
        Map coordinates from cropped region back to global image

        Args:
            local_x, local_y: coordinates in the cropped search region

        Returns:
            global_x, global_y: coordinates in the global image
        """
        if hasattr(self, 'crop_offset'):
            offset_x, offset_y = self.crop_offset
            return local_x + offset_x, local_y + offset_y
        else:
            return local_x, local_y


class VideoDataset2():
    def __init__(self, source_img, tmp_vid_src, tmp_size,thres=0.7, transform=None,half=False,skip_frames=1,frame_rate=1):
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        self.half = half
        self.image_name = source_img
        self.image_raw = cv2.imread(self.image_name)
        self.thresh = thres
        self.image = self.transform(self.image_raw).unsqueeze(0)
        self.vc = cv2.VideoCapture(tmp_vid_src)
        if not self.vc.isOpened(): print("Unable to open video source")
        for i in range(skip_frames):
            self.vc.read()
        self.tmp_size = tmp_size
        if self.half:
            self.image = self.image.half()
        self.skip = round(self.vc.get(5) / frame_rate)

    def get_next_image(self):
        for _ in range(self.skip):
            self.vc.read()
        if self.vc.isOpened():
            ret, img = self.vc.read()
            if ret:
                template = cv2.resize(img, self.tmp_size)
                img = template.copy()
                if self.transform:
                    template = self.transform(template)
                if self.half:
                    template = template.half()
                return {'image': self.image,
                        'image_raw': self.image_raw,
                        'image_name': self.image_name,
                        'template': template.unsqueeze(0),
                        'template_h': template.size()[-2],
                        'template_w': template.size()[-1],
                        'thresh': self.thresh,
                        'img':img}
        return None


class Featex():
    def __init__(self, model, use_cuda, save_features=None):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.feature1 = save_features
        self.feature2 = save_features
        self.model = copy.deepcopy(model.eval())
        self.model = self.model[:17]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.model = self.model.cuda()
        else:
            print("CUDA not available, running on CPU.")
        self.model[2].register_forward_hook(self.save_feature1)
        self.model[16].register_forward_hook(self.save_feature2)

    def save_feature1(self, module, input, output):
        self.feature1 = output.detach()
        print("Shape of feature1 before resizing:", self.feature1.shape)

    def save_feature2(self, module, input, output):
        self.feature2 = output.detach()
        print("Shape of feature2 before resizing:", self.feature2.shape)

    def __call__(self, input, mode='big'):
        if self.use_cuda:
            input = input.cuda()
        _ = self.model(input)
        print("Shape of feature1 before resizing:", self.feature1.shape)
        print("Shape of feature2 before resizing:", self.feature2.shape)
        if mode == 'big':
            self.feature1 = F.interpolate(self.feature1, size=(self.feature2.size()[2], self.feature2.size()[3]), mode='bilinear', align_corners=True)
        else:
            self.feature2 = F.interpolate(self.feature2, size=(self.feature1.size()[2], self.feature1.size()[3]), mode='bilinear', align_corners=True)
        print("Shape of feature1 after resizing:", self.feature1.shape)
        print("Shape of feature2 after resizing:", self.feature2.shape)
        return torch.cat((self.feature1, self.feature2), dim=1)


def MyNormLayer(x1, x2):
    bs, _, H, W = x1.size()
    _, _, h, w = x2.size()
    x1 = x1.view(bs, -1, H * W)
    x2 = x2.view(bs, -1, h * w)
    concat = torch.cat((x1, x2), dim=2)
    x_mean = torch.mean(concat, dim=2, keepdim=True)
    x_std = torch.std(concat, dim=2, keepdim=True)
    x1 = (x1 - x_mean) / x_std
    x2 = (x2 - x_mean) / x_std
    x1 = x1.view(bs, -1, H, W)
    x2 = x2.view(bs, -1, h, w)
    return [x1, x2]


def MyNormLayer_new(x1):
    bs, _, h,w = x1.size()
    x1 = x1.view(bs, -1, h * w)
    x_mean=x1.mean()
    x_std =x1.std()
    x1=(x1-x_mean)/x_std
    x1=x1.view(bs,-1,h,w)
    return x1


class CreateModel():
    def __init__(self, alpha, model, use_cuda):
        self.alpha = alpha
        self.featex = Featex(model, use_cuda)
        self.I_feat = None

    def __call__(self, template, image):
        print("Calculating feature map for template...")
        T_feat = self.featex(template)
        print("Calculating feature map for source image...")
        self.I_feat = self.featex(image)
        conf_maps = None
        batchsize_T = T_feat.size()[0]
        for i in range(batchsize_T):
            T_feat_i = T_feat[i].unsqueeze(0)
            I_feat_norm, T_feat_i = MyNormLayer(self.I_feat, T_feat_i)
            dist = torch.einsum("xcab,xcde->xabde", I_feat_norm / torch.norm(I_feat_norm, dim=1, keepdim=True),
                                T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True))
            conf_map = QATM(self.alpha)(dist)
            if conf_maps is None:
                conf_maps = conf_map
            else:
                conf_maps = torch.cat([conf_maps, conf_map], dim=0)
        return conf_maps


class CreateModel_2():
    def __init__(self, alpha, model, use_cuda,image):
        self.alpha = alpha
        self.featex = Featex(model, use_cuda,save_features=True)
        self.I_feat = None
        self.I_feat = self.featex(image)
        self.I_feat_norm = MyNormLayer_new(self.I_feat)
        self.I_feat_torch_norm = torch.norm(self.I_feat_norm, dim=1, keepdim=True)

    def __call__(self, template, image):
        T_feat = self.featex(template)
        conf_maps = None
        batchsize_T = T_feat.size()[0]
        for i in range(batchsize_T):
            T_feat_i = T_feat[i].unsqueeze(0)
            T_feat_i = MyNormLayer_new(T_feat_i)
            dist = torch.einsum("xcab,xcde->xabde", self.I_feat_norm / self.I_feat_torch_norm,
                                T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True))
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
        confidence = torch.sqrt(F.softmax(self.alpha * xm_ref, dim=1) * F.softmax(self.alpha * xm_qry, dim=2))
        conf_values, ind3 = torch.topk(confidence, 1)
        ind1, ind2 = torch.meshgrid(torch.arange(batch_size), torch.arange(ref_row * ref_col))
        ind1 = ind1.flatten()
        ind2 = ind2.flatten()
        ind3 = ind3.flatten()
        if x.is_cuda:
            ind1 = ind1.cuda()
            ind2 = ind2.cuda()
        values = confidence[ind1, ind2, ind3]
        values = torch.reshape(values, [batch_size, ref_row, ref_col, 1])
        return values

    def compute_output_shape(self, input_shape):
        bs, H, W, _, _ = input_shape
        return (bs, H, W, 1)


def nms(score, w_ini, h_ini, thresh=0.7):
    dots = np.array(np.where(score > thresh * score.max()))
    x1 = dots[1] - w_ini // 2
    x2 = x1 + w_ini
    y1 = dots[0] - h_ini // 2
    y2 = y1 + h_ini
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = score[dots[0], dots[1]]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= 0.5)[0]
        order = order[inds + 1]
    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2, 0, 1)
    return boxes


def plot_result(image_raw, boxes, show=False, save_name=None, color=(0, 255, 0),text=None, text_loc=None):
    d_img = image_raw.copy()
    for box in boxes:
        x = (box[0][0] + box[1][0]) // 2
        y = (box[0][1] + box[1][1]) // 2
        d_img = cv2.rectangle(d_img, (box[0][0]-5, box[0][1]-5),(box[0][0]+5, box[0][1]+5) , (255,255,255), 2)
        if text:
            d_img = cv2.putText(d_img,str(text),(box[0][0]+5,box[0][1]-5),cv2.FONT_HERSHEY_DUPLEX,0.4,(255,255,255),1)
    if show:
        import matplotlib.pyplot as plt
        plt.imshow(d_img)
    if save_name:
        cv2.imwrite(save_name, d_img[:, :, ::-1])
    return d_img


def nms_multi(scores, w_array, h_array, thresh_list, multibox=True):
    indices = np.arange(scores.shape[0])
    maxes = np.max(scores.reshape(scores.shape[0], -1), axis=1)
    scores_omit = scores[maxes > 0.1 * maxes.max()]
    indices_omit = indices[maxes > 0.1 * maxes.max()]
    dots = None
    dots_indices = None
    for index, score in zip(indices_omit, scores_omit):
        dot = np.array(np.where(score > thresh_list[index] * score.max()))
        if dots is None:
            dots = dot
            dots_indices = np.ones(dot.shape[-1]) * index
        else:
            dots = np.concatenate([dots, dot], axis=1)
            dots_indices = np.concatenate([dots_indices, np.ones(dot.shape[-1]) * index], axis=0)
    dots_indices = dots_indices.astype(np.int64)
    x1 = dots[1] - w_array[dots_indices] // 2
    x2 = x1 + w_array[dots_indices]
    y1 = dots[0] - h_array[dots_indices] // 2
    y2 = y1 + h_array[dots_indices]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = scores[dots_indices, dots[0], dots[1]]
    order = scores.argsort()[::-1]
    dots_indices = dots_indices[order]
    keep = []
    keep_index = []
    if multibox:
        while order.size > 0:
            i = order[0]
            index = dots_indices[0]
            keep.append(i)
            keep_index.append(index)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= 0.05)[0]
            order = order[inds + 1]
            dots_indices = dots_indices[inds + 1]
    else:
        i = order[0]
        index = dots_indices[0]
        keep.append(i)
        keep_index.append(index)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= 0.05)[0]
        order = order[inds + 1]
        dots_indices = dots_indices[inds + 1]
    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2, 0, 1)
    return boxes, np.array(keep_index)


def plot_result_multi(image_raw, boxes, indices, show=True, save_name=None, color_list=None):
    d_img = image_raw.copy()
    if color_list is None:
        color_list = color_palette("hls", indices.max() + 1)
        color_list = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color_list))
    for i in range(len(indices)):
        d_img = plot_result(d_img, boxes[i][None, :, :].copy(), color=color_list[indices[i]])
    if show:
        import matplotlib.pyplot as plt
        plt.imshow(d_img)
    if save_name:
        cv2.imwrite(save_name, d_img[:, :, ::-1])
    return d_img


def run_one_sample(model, template, image):
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
        score = compute_score(gray, w, h)
        score[score > -1e-7] = score.min()
        score = np.exp(score / (h * w))
        scores.append(score)
    return np.array(scores)


def run_one_sample_2(model, template, image):
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
        score = compute_score(gray, w, h)
        score[score > -1e-7] = score.min()
        score = np.exp(score / (h * w))
        scores.append(score)
    return np.array(scores)


def run_multi_sample(model, dataset):
    scores = []
    w_array = []
    h_array = []
    thresh_list = []
    for data in dataset:
        score = run_one_sample(model, data['template'], data['image'])
        scores.append(score)
        w_array.append(data['template_w'])
        h_array.append(data['template_h'])
        thresh_list.append(data['thresh'])
    return np.squeeze(np.array(scores), axis=1), np.array([w_array]), np.array([h_array]), thresh_list


model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=25, use_cuda=True)
