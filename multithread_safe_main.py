from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime

import cv2
import numpy as np
from typing import List

from main import video_to_frames

# ---------- 线程局部存储：每线程一份 OCR ----------
_thread_local = threading.local()

def _get_paddle_model():
    """线程安全地返回 PaddleOCR 实例（无则新建）"""
    if not hasattr(_thread_local, "model"):
        from paddleocr import PaddleOCR
        _thread_local.model = PaddleOCR(use_angle_cls=True, lang="ch")  # 或 'en'
    return _thread_local.model

def _paddle_run(frame: np.ndarray) -> str:
    model = _get_paddle_model()
    result = model.ocr(frame, cls=True)
    return "\n".join(line[1][0] for line in result[0]) if result else ""


def ocr_frames_threadsafe(frames: List[np.ndarray],
                          max_workers: int | None = None) -> List[str]:
    """
    多线程 PaddleOCR（每线程独立模型实例）
    """
    if max_workers is None:
        # CPU 核心数 - 1 会比较稳；GPU 场景设 1~2
        import os
        max_workers = 2

    texts = [None] * len(frames)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut2idx = {pool.submit(_paddle_run, f): i
                   for i, f in enumerate(frames)}
        for fut in as_completed(fut2idx):
            texts[fut2idx[fut]] = fut.result()
    return texts

if __name__ == '__main__':
    frames = video_to_frames("short.mp4")  # 你的抽帧函数
    print('paddle--------------------------------------------')
    start = datetime.now()
    texts = ocr_frames_threadsafe(frames, max_workers=4)
    print("time used: ")
    print(datetime.now() - start)

    print(texts[:3])