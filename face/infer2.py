from facenet_pytorch import InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import cv2
import pandas as pd
import os
from torchvision.transforms import functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from utils.nms.py_cpu_nms import py_cpu_nms
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer

from torchvision import transforms
import cv2
from PIL import Image


workers = 0 if os.name == 'nt' else 4

trained_model = "weights/mobilenet0.25_Final.pth"
network = 'mobile0.25'
dataset = 'FDDB'
confidence_threshold = 0.02
nms_threshold = 0.4,
save_image = True
vis_thres = 0.5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}
_t = {'forward_pass': Timer(), 'misc': Timer()}


def load_model(net, pretrained_path, load_to_cpu=False):
    print('Loading pretrained model from {}'.format(pretrained_path))

    def remove_prefix(state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def check_keys(model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location="cpu")
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(net, pretrained_dict)
    net.load_state_dict(pretrained_dict, strict=False)
    return net

def load_models():
    torch.set_grad_enabled(False)
    net = RetinaFace(cfg=cfg_mnet, phase = 'test')
    net = load_model(net, trained_model)
    print('Finished loading retinaface model!')
    #cudnn.benchmark = True
    net = net.eval().to(device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print('Finished loading resnet model!')
    return net, resnet


net, resnet = load_models()

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def get_image(path, trans):
    img = Image.open(path)
    img = trans(img)
    return img


trans_cropped = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        out = cv2.resize(
            img[box[1]:box[3], box[0]:box[2]],
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out


def save_img(img, path):
    if isinstance(img, np.ndarray):
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img.save(path)


def get_size(img):
    if isinstance(img, np.ndarray):
        return img.shape[1::-1]
    else:
        return img.size


def extract_face(img, box, image_size=160, margin=0, save_path=None):

    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)

    face = F.to_tensor(np.float32(face))
    face = fixed_image_standardization(face)
    return face


def box_handle(img, conf, im_height, im_width, scale, loc, landms):
    priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]
    dets = np.concatenate((dets, landms), axis=1)
    return dets


def img_hanlder(img):
    img = np.float32(img)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    return img, scale, im_height, im_width


def _get_embeddings(aligned, names=None):
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cuda()
    return embeddings


def cal_angle(b):
    eye_left_x, eye_left_y, eye_right_x, eye_right_y, nose_x, nose_y = b[5], b[6], b[7], b[8], b[9], b[10]
    maxy = max(eye_right_y, eye_left_y)
    miny = min(eye_right_y, eye_left_y)
    if maxy < nose_y:
        return 0
    if miny > nose_y:
        return 180
    if eye_left_y > eye_right_y:
        return 90
    if eye_right_y > eye_left_y:
        return -90


def get_aligned_file(x):
    img, scale, im_height, im_width = img_hanlder(x)
    _t['forward_pass'].tic()
    loc, conf, landms = net(img)  # forward pass
    _t['forward_pass'].toc()
    _t['misc'].tic()
    dets = box_handle(img, conf, im_height, im_width, scale, loc, landms)
    _t['misc'].toc()
    i = 0 
    num_images = 1
    print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images,
                                                                                  _t['forward_pass'].average_time,
                                                                                  _t['misc'].average_time))
    b = dets[0]
    aligned = None
    if b[4] >= vis_thres:
        b = list(map(int, b))
        aligned = extract_face(x, b)
        angle = cal_angle(b)
    return aligned


def get_embeddings(img):
    aligned = get_aligned_file(img)
    if aligned is None:
        return []
    embeddings = _get_embeddings([aligned]).cpu()[0].numpy()
    return embeddings

