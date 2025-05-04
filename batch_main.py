import datetime

from paddleocr import PaddleOCR
import easyocr, cv2
from main import paddle_model,easyocr_model

def paddle_batch(frames, batch_size=8):
    """
    按 batch_size 分批 OCR，返回文本列表
    """
    out = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        # 直接传 list[np.ndarray]
        results = paddle_model.ocr(batch, cls=True,det=False)      # list[list[line]]
        out.extend(
            "\n".join(line[1][0] for line in r[0]) if r else ""
            for r in results
        )
    return out


def easyocr_batch(frames, batch_size=8):
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    return easyocr_model.readtext_batched(
        rgb_frames, batch_size=batch_size, detail=0
    )  # list[str]


if __name__ == '__main__':
    from main import video_to_frames

    all_frames = video_to_frames("short.mp4")
    print('paddle--------------------------------------------')
    start = datetime.datetime.now()
    paddle_batch(all_frames)
    print("time used: ")
    print(datetime.datetime.now() - start)
    print('easyocr--------------------------------------------')
    start = datetime.datetime.now()
    easyocr_batch(all_frames)
    print("time used: ")
    print(datetime.datetime.now() - start)
