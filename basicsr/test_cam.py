import logging
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str
from basicsr.models.grad_cam import GradCAM
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def pad_test(img, window_size):
    mod_pad_h, mod_pad_w = 0, 0
    _, _, h, w = img.size()
    if h % window_size != 0:
        mod_pad_h = window_size - h % window_size
    if w % window_size != 0:
        mod_pad_w = window_size - w % window_size
    img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return img

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == "train":
            continue
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = create_model(opt).net_g
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    target_layer = model.decoder_level1[0]
    gradcam = GradCAM(model, target_layer)

    img = cv2.imread("G:\dataset\stamp-test\seal_datasets_250111\\test\extract_img_stamp/3.png")
    x = img2tensor(img, bgr2rgb=True, float32=True)
    normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
    x = x.unsqueeze(0).to(device)
    # x = x.to(device)
    x = pad_test(x, 8)

    heatmap = gradcam(x)

    img = cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    result = overlay_heatmap(img, heatmap)
    cv_show(result)


def cv_show(img):
    """
    临时显示图片
    @param img: 图片
    @return:
    """
    cv2.imshow("temp", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def overlay_heatmap(img, heatmap, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1-alpha, 0)
    return superimposed_img



if __name__ == '__main__':
    main()
