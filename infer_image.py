import cv2
import dlib
import onnxruntime
import numpy as np
from face_swapper.utils import paste_back, align_crop
from face_swapper import ArcFace, Ghost, SimSwap, SimSwapUnofficial

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_kps(img):
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_detector(image_gray)[0]
    shape = shape_predictor(image_gray, face)
    left_eye = np.mean(shape.parts()[36:42], axis=0)
    right_eye = np.mean(shape.parts()[42:48], axis=0)
    nose = shape.part(30)
    left_mouth = shape.part(48)
    right_mouth = shape.part(54)
    kps = [left_eye, right_eye, nose, left_mouth, right_mouth]
    kps = np.array([[lmk.x, lmk.y] for lmk in kps]).astype('float32')
    return kps

onnx_params = {
    'sess_options': onnxruntime.SessionOptions(),
    'providers': ["CUDAExecutionProvider", "CPUExecutionProvider"]
}

# ghost_path = "ghost_unet_1_block.onnx"
ghost_path = "ghost_unet_2_block.onnx"
# ghost_path = "ghost_unet_3_block.onnx"
swapper = Ghost(ghost_path, **onnx_params)
ghost_backbone_path = "ghost_arcface_backbone.onnx"
backbone = ArcFace(ghost_backbone_path, **onnx_params)

# simswap_path = "simswap.onnx"
# simswap_path = "simswap_512_beta.onnx"
# swapper = SimSwap(simswap_path, **onnx_params)
# simswap_backbone_path = "simswap_arcface_backbone.onnx"
# backbone = ArcFace(simswap_backbone_path, **onnx_params)

source = cv2.imread("example/source.jpg")
target = cv2.imread("example/target.jpg")
mask = cv2.imread("example/mask.jpg")

source_embedding = backbone.forward(source, get_kps(source))
target_cropped, matrix = align_crop(target, get_kps(target), swapper.align_crop_size, mode=swapper.align_crop_mode)
swap_face = swapper.forward(target_cropped, source_embedding)
mask = cv2.resize(mask, (swap_face.shape[1], swap_face.shape[0])).astype('float32') / 255
matrix *= (swap_face.shape[0] / swapper.align_crop_size)
final_image = paste_back(target, swap_face, mask, matrix)
cv2.imwrite("example/swapped.jpg", final_image)