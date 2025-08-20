'''
@Author  ：zww
@Date    ：2024/1/5 15:47 
@Description : *
'''
import os
import onnxruntime
import cv2
import numpy as np
from skimage import img_as_ubyte
import timeit

dir_cur = os.path.dirname(os.path.abspath(__file__))
default_model_path = os.path.join(dir_cur, 'Deseal/pretrained_models/restormer.onnx') .pth
print("available_providers: " + str(onnxruntime.get_available_providers()))
def load_model(model_path=default_model_path, device='gpu', gpu_mem_limit=2):
    # Convert onnxruntime version to number
    ort_version = onnxruntime.__version__
    snd_point_pos = ort_version.find('.', ort_version.find('.') + 1)
    if snd_point_pos > 2:
        ort_version = ort_version[:snd_point_pos]
    ort_version = eval(ort_version)

    available_providers = onnxruntime.get_available_providers()
    if device == 'gpu' and not available_providers.__contains__('CUDAExecutionProvider'):
        # print("您的设备不支持GPU，已为您切换至CPU")
        device = "cpu"

    if ort_version >= 1.6:
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': gpu_mem_limit * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ] if device != 'cpu' else ['CPUExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
    print('Current onnxruntime version: ', onnxruntime.__version__)
    print('Current device: ', device)
    print(providers)
    providers = onnxruntime.get_available_providers()
    model = onnxruntime.InferenceSession(model_path, providers=providers)

    return model



img_multiple_of = 8

def predict(net, image_list):
    result = []
    for image in image_list:
        starttime = timeit.default_timer()
        img = image

        temp = img.astype(float)
        temp = np.divide(temp, 255.).transpose((2, 0, 1))
        temp = np.expand_dims(temp, 0)

        height,width = temp.shape[2], temp.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        temp = np.pad(temp, ((0,0),(0,0),(0,padh),(0,padw)), mode='reflect')
        temp = temp.astype('float32')
        # 准备输入信息
        outputs = net.run(None, {net.get_inputs()[0].name: temp})[0]
        # pred = net.run([net.get_outputs()[0].name], {net.get_inputs()[0].name: temp})[0]
        outputs = np.clip(outputs, 0, 1)
        outputs = outputs[:, :, :height, :width]
        outputs = outputs.transpose((0, 2, 3, 1))

        restored = img_as_ubyte(outputs[0])
        result.append(cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
        endtime = timeit.default_timer()
        print("Removing Watermark  Consume: " + str(endtime - starttime) + " s")

    return result

