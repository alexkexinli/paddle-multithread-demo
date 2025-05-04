import datetime

import cv2, numpy as np, threading, queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from paddleocr import PaddleOCR

from main import video_to_frames

# -------------------------------------------------
# 1. 预创建 OCR 对象并放进队列（Pool）
# -------------------------------------------------
N_OCR   = 6                                   # 显存够：多放几个
OCR_OPTS = dict(use_angle_cls=True, lang="ch") # 'en' or others

_ocr_queue: "queue.Queue[PaddleOCR]" = queue.Queue(maxsize=N_OCR)
for _ in range(N_OCR):
    _ocr_queue.put(PaddleOCR(**OCR_OPTS))

# -------------------------------------------------
# 2. 上下文管理器：安全借还
# -------------------------------------------------
from contextlib import contextmanager

@contextmanager
def acquire_ocr() -> PaddleOCR:                # type: ignore[override]
    ocr = _ocr_queue.get()                     # 取（阻塞直到有空位）
    try:
        yield ocr
    finally:
        _ocr_queue.put(ocr)                    # 归还

# -------------------------------------------------
# 3. 单帧 OCR Worker
# -------------------------------------------------
def _ocr_one(frame: np.ndarray) -> str:
    with acquire_ocr() as ocr:
        result = ocr.ocr(frame, cls=True)
        return "\n".join(l[1][0] for l in result[0]) if result else ""

# -------------------------------------------------
# 4. 批量并发接口
# -------------------------------------------------
def ocr_frames_pool(frames: List[np.ndarray],
                    max_workers: int | None = None) -> List[str]:
    if max_workers is None:
        import os
        max_workers = os.cpu_count() or 4      # 按需调整

    texts: List[str] = [None] * len(frames)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut2idx = {pool.submit(_ocr_one, f): i for i, f in enumerate(frames)}
        for fut in as_completed(fut2idx):
            texts[fut2idx[fut]] = fut.result()
    return texts

# --------------- 示例 -----------------
if __name__ == "__main__":
    all_frames = video_to_frames("short.mp4")
    print('paddle--------------------------------------------')
    start = datetime.datetime.now()
    texts = ocr_frames_pool(all_frames, max_workers=8)
    print("time used: ")
    print(datetime.datetime.now() - start)


    print(texts[:3])
