import datetime

from paddleocr import PaddleOCR
import easyocr, cv2

from concurrent.futures import ThreadPoolExecutor, as_completed

def ocr_frames_threadpool(frames, ocr_type="paddle", max_workers=4):
    # 把前面 ocr_frames 里的 instantiation 拆出来，确保只初始化一次
    ocr_type = ocr_type.lower()
    if ocr_type == "paddle":
        from paddleocr import PaddleOCR
        model = PaddleOCR(use_angle_cls=True, lang="ch")
        def _run(img): return "\n".join(l[1][0] for l in model.ocr(img, cls=True)[0]) if model else ""
    else:
        import easyocr, cv2
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        def _run(img): return "\n".join(reader.readtext(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), detail=0))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run, f): i for i, f in enumerate(frames)}
        texts = [None] * len(frames)
        for fut in as_completed(futures):
            idx = futures[fut]
            texts[idx] = fut.result()
    return texts



if __name__ == '__main__':
    from main import video_to_frames

    all_frames = video_to_frames("short.mp4")
    print('paddle--------------------------------------------')
    start = datetime.datetime.now()
    ocr_frames_threadpool(all_frames[:10],"paddle",4)
    print("time used: ")
    print(datetime.datetime.now() - start)
    # print('easyocr--------------------------------------------')
    # start = datetime.datetime.now()
    # ocr_frames_threadpool(all_frames, "easyocr", 10)
    # print("time used: ")
    # print(datetime.datetime.now() - start)
