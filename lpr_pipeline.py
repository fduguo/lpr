import cv2
import math
import copy
import numpy as np
import onnxruntime as ort
from pathlib import Path
import os


# ─────────────────────────────────────────────
# 字符集
# ─────────────────────────────────────────────
TOKEN = ['blank', "'", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
         'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
         '云', '京', '冀', '吉', '学', '宁', '川', '挂', '新', '晋', '桂',
         '民', '沪', '津', '浙', '渝', '港', '湘', '琼', '甘', '皖', '粤',
         '航', '苏', '蒙', '藏', '警', '豫', '贵', '赣', '辽', '鄂', '闽',
         '陕', '青', '鲁', '黑', '领', '使', '澳']

# ─────────────────────────────────────────────
# 检测预处理 / 后处理
# ─────────────────────────────────────────────
def letter_box(img, size=(320, 320)):
    h, w, _ = img.shape
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    top  = (size[0] - new_h) // 2
    left = (size[1] - new_w) // 2
    img_resized = cv2.resize(img, (new_w, new_h))
    img_padded  = cv2.copyMakeBorder(img_resized, top, size[0]-new_h-top,
                                     left, size[1]-new_w-left,
                                     cv2.BORDER_CONSTANT, value=(0,0,0))
    return img_padded, r, left, top

def detect_preprocess(img, size=(320, 320)):
    padded, r, left, top = letter_box(img, size)
    x = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return x[np.newaxis], r, left, top

def xywh2xyxy(boxes):
    b = copy.deepcopy(boxes)
    b[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    b[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    b[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    b[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return b

def nms(boxes, iou_thresh=0.5):
    idx = np.argsort(boxes[:, 4])[::-1]
    keep = []
    while idx.size > 0:
        i = idx[0]; keep.append(i)
        x1 = np.maximum(boxes[i,0], boxes[idx[1:],0])
        y1 = np.maximum(boxes[i,1], boxes[idx[1:],1])
        x2 = np.minimum(boxes[i,2], boxes[idx[1:],2])
        y2 = np.minimum(boxes[i,3], boxes[idx[1:],3])
        inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
        union = ((boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1]) +
                 (boxes[idx[1:],2]-boxes[idx[1:],0])*(boxes[idx[1:],3]-boxes[idx[1:],1]))
        iou = inter / (union - inter)
        idx = idx[np.where(iou <= iou_thresh)[0] + 1]
    return keep

def detect_postprocess(dets, r, left, top, conf_thresh=0.25, iou_thresh=0.5):
    choice = dets[:, :, 4] > conf_thresh
    dets = dets[choice]
    if len(dets) == 0:
        return []
    dets[:, 13:15] *= dets[:, 4:5]
    boxes  = xywh2xyxy(dets[:, :4])
    score  = np.max(dets[:, 13:15], axis=-1, keepdims=True)
    index  = np.argmax(dets[:, 13:15], axis=-1).reshape(-1, 1)
    output = np.concatenate((boxes, score, dets[:, 5:13], index), axis=1)
    output = output[nms(output, iou_thresh)]
    # 还原坐标
    output[:, [0,2,5,7,9,11]] -= left
    output[:, [1,3,6,8,10,12]] -= top
    output[:, [0,2,5,7,9,11]] /= r
    output[:, [1,3,6,8,10,12]] /= r
    return output

# ─────────────────────────────────────────────
# 透视变换校正
# ─────────────────────────────────────────────
def get_rotate_crop_image(img, points):
    points = points.astype(np.float32)
    w = max(int(np.linalg.norm(points[0]-points[1])),
            int(np.linalg.norm(points[2]-points[3])))
    h = max(int(np.linalg.norm(points[0]-points[3])),
            int(np.linalg.norm(points[1]-points[2])))
    dst = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(points, dst)
    return cv2.warpPerspective(img, M, (w, h))

# ─────────────────────────────────────────────
# 识别预处理
# ─────────────────────────────────────────────
def encode_image(img, target_h=48, target_w=160):
    h, w, _ = img.shape
    ratio    = w / float(h)
    new_w    = min(max(int(target_h * ratio), 48), target_w)
    resized  = cv2.resize(img, (new_w, target_h)).astype(np.float32)
    resized  = (resized.transpose(2, 0, 1) - 127.5) / 127.5
    canvas   = np.zeros((3, target_h, target_w), dtype=np.float32)
    canvas[:, :, :new_w] = resized
    return canvas[np.newaxis]

# ─────────────────────────────────────────────
# CTC解码
# ─────────────────────────────────────────────
def ctc_decode(output, token_list):
    prod    = output[0]                        # (40, vocab)
    indices = np.argmax(prod, axis=2)[0]       # (40,)
    probs   = np.max(prod, axis=2)[0]
    chars, confs = [], []
    for i, idx in enumerate(indices):
        if idx == 0: continue
        if i > 0 and indices[i-1] == idx: continue
        chars.append(token_list[int(idx)])
        confs.append(probs[i])
    return ''.join(chars), float(np.mean(confs)) if confs else 0.0

# ─────────────────────────────────────────────
# 主推理类
# ─────────────────────────────────────────────
class LPRPipeline:
    def __init__(self, model_dir: str, detect_level: str = 'low'):
        model_dir = Path(model_dir)
        det_name  = 'y5fu_320x_sim.onnx' if detect_level == 'low' else 'y5fu_640x_sim_fixed.onnx'
        self.det_size = (320, 320) if detect_level == 'low' else (640, 640)

        self.det = ort.InferenceSession(str(model_dir / det_name),
                                        providers=['CPUExecutionProvider'])
        self.rec = ort.InferenceSession(str(model_dir / 'rpv3_mdict_160_r3_fixed.onnx'),
                                        providers=['CPUExecutionProvider'])
        self.cls = ort.InferenceSession(str(model_dir / 'litemodel_cls_96x_r1_fixed.onnx'),
                                        providers=['CPUExecutionProvider'])
        self.det_in  = self.det.get_inputs()[0].name
        self.det_out = self.det.get_outputs()[0].name
        self.rec_in  = self.rec.get_inputs()[0].name
        self.rec_out = self.rec.get_outputs()[0].name

        # 字符集
        self.token = TOKEN

    def __call__(self, img: np.ndarray):
        """
        输入: BGR图像 (np.ndarray)
        输出: list of dict，每个元素包含 plate/confidence/box/landmarks
        """
        # 1. 检测
        x, r, left, top = detect_preprocess(img, self.det_size)
        raw = self.det.run([self.det_out], {self.det_in: x})[0]
        detections = detect_postprocess(raw, r, left, top)
        if len(detections) == 0:
            return []

        results = []
        for det in detections:
            rect       = det[:4].astype(int)
            score      = float(det[4])
            landmarks  = det[5:13].reshape(4, 2).astype(int)
            layer_num  = int(det[13])   # 0=单层 1=双层

            # 2. 透视校正
            crop = get_rotate_crop_image(img, landmarks.astype(np.float32))

            # 3. 识别
            if layer_num == 1:  # 双层车牌
                h = crop.shape[0]
                line = int(h * 0.4)
                top_code,    top_conf    = self._recognize(crop[:line])
                bottom_code, bottom_conf = self._recognize(crop[line:])
                plate_code = top_code + bottom_code
                confidence = (top_conf + bottom_conf) / 2
            else:
                plate_code, confidence = self._recognize(crop)

            if len(plate_code) < 7:
                continue

            results.append({
                'plate':      plate_code,
                'confidence': round(confidence, 4),
                'box':        rect.tolist(),
                'landmarks':  landmarks.tolist(),
                'det_score':  round(score, 4),
            })

        return results

    def _recognize(self, crop):
        x   = encode_image(crop)
        out = self.rec.run([self.rec_out], {self.rec_in: x})
        return ctc_decode(out, self.token)


# ─────────────────────────────────────────────
# 测试入口
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    MODEL_DIR = r'D:\Users\Lenovo\.hyperlpr3\20230229\onnx'
    IMAGE     = sys.argv[1] if len(sys.argv) > 1 else 'input.png'

    pipeline = LPRPipeline(MODEL_DIR)
    img      = cv2.imread(IMAGE)
    assert img is not None, f"无法读取图片: {IMAGE}"

    results = pipeline(img)
    print(f"\n共检测到 {len(results)} 个车牌：")
    for r in results:
        print(f"  车牌号: {r['plate']}  置信度: {r['confidence']}  框: {r['box']}")

    # 画框保存
    out = img.copy()
    for r in results:
        x1,y1,x2,y2 = r['box']
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(out, r['plate'], (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    out_path = IMAGE.replace('.png','_result.png').replace('.jpg','_result.jpg')
    cv2.imwrite(out_path, out)
    print(f"结果图保存到: {out_path}")