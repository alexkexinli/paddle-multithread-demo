import os, sys
cuda = os.getenv("CUDA_PATH")
if cuda:
    os.add_dll_directory(os.path.join(cuda, "bin"))

import cv2
import numpy as np
from pathlib import Path
from memory_profiler import profile
import datetime

import torch
import torch, platform, sys
print("torch :", torch.__version__)
print("python:", sys.version.split()[0])
print("cuda  :", torch.version.cuda)


print(torch.cuda.is_available())
import paddle
gpu_available  = paddle.device.is_compiled_with_cuda()
print("GPU available:", gpu_available)
from paddleocr import PaddleOCR


import easyocr

paddle_model = PaddleOCR(use_angle_cls=True, lang="ch")  # or 'en'
easyocr_model = easyocr.Reader(['ch_sim', 'en'], gpu=True)  # GPU=True 自动用 CUDA


@profile
def ocr_frames(frames: list[np.ndarray], ocr_type: str = "paddle") -> list[str]:
    """
    对视频帧批量 OCR

    Parameters
    ----------
    frames : list[np.ndarray]
        BGR 彩色帧列表（cv2 默认格式）
    ocr_type : {"paddle", "easyocr"}
        选择底层 OCR 引擎（大小写不敏感）

    Returns
    -------
    list[str]
        OCR 结果列表，每个元素对应 frames 同索引的文本
    """
    ocr_type = ocr_type.lower()
    if ocr_type == "paddle":
        # -------- PaddleOCR 初始化（一次即可） --------

        # use_angle_cls=True 可自动旋转；根据语言换 lang
        def run_ocr(img):
            # PaddleOCR 返回 [ [ [x1,y1], ... ], (text, score) ], per line
            result = paddle_model.ocr(img, cls=True)

            # 拼接所有行文本，保留顺序
            return "\n".join([line[1][0] for line in result[0]]) if result else ""
    elif ocr_type == "easyocr":
        # -------- EasyOCR 初始化 --------

        def run_ocr(img):
            # EasyOCR 输入是 RGB，需要转换
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = easyocr_model.readtext(rgb, detail=0,batch_size=10)  # detail=0 → 仅文本
            return "\n".join(result)
    else:
        raise ValueError("ocr_type must be 'paddle' or 'easyocr'")

    # -------- 主循环 --------
    texts: list[str] = []
    for frame in frames:
        try:
            text = run_ocr(frame)
            print("current image text:"+text)
        except Exception as e:
            text = f"[ERROR] {e}"
        texts.append(text)

    return texts


@profile
def video_to_frames(path: str | Path,
                    resize_wh: tuple[int, int] | None = None,
                    ):
    """
    读取视频，每秒仅抽取 1 帧，返回 ndarray 或生成器
    -------------------------------------------------
    参数
    ----
    path : str | Path
        视频文件路径
    resize_wh : (width, height) or None
        是否等比缩放到指定大小；None 保持原尺寸
    as_generator : bool
        True 则返回生成器（yield），False 返回 np.ndarray

    返回
    ----
    np.ndarray shape = (N, H, W, 3)   或   生成器
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频：{path}")

    fps = cap.get(cv2.CAP_PROP_FPS)          # 帧/秒
    step = int(round(fps)) or 1              # 每秒取 1 帧 ⇒ 每 step 取 1 帧
    frames = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:                          # 读完了
            break

        if idx % step == 0:                  # 只保留关键帧
            if resize_wh is not None:
                frame = cv2.resize(frame, resize_wh,
                                   interpolation=cv2.INTER_AREA)
            # OpenCV 默认是 BGR，可按需 cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)         # 累积到列表

        idx += 1

    cap.release()

    return frames

# ------------------ 用法示例 ------------------
if __name__ == "__main__":
    # 方式一：一次性返回 ndarray
    all_frames = video_to_frames("short.mp4")
    total_bytes = sum(f.nbytes for f in all_frames)  # 每张 ndarray 的字节数
    total_mb = total_bytes / 1024 / 1024
    print(f"抽取到 {len(all_frames)} 帧，总内存约 {total_mb:.2f} MB")

    print('paddle--------------------------------------------')
    start = datetime.datetime.now()
    paddle_texts = ocr_frames(all_frames, "paddle")  # PaddleOCR
    print("time used: ")
    print(datetime.datetime.now()-start)

    # print('easyocr--------------------------------------------')
    # start = datetime.datetime.now()
    # easy_texts = ocr_frames(all_frames[:10], "easyocr")  # EasyOCR
    # print("time used: ")
    # print(datetime.datetime.now() - start)
    # for i, txt in enumerate(easy_texts[:10]):
    #     print(f"Frame {i}:")
    #     print(txt)
    #     print("-" * 20)



