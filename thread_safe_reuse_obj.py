
ocr1 = PaddleOCR(lang='en', use_gpu=True)
ocr2 = PaddleOCR(lang='en', use_gpu=True)
ocr3 = PaddleOCR(lang='en', use_gpu=True)

parallel_thread_counter = 0
parallel_thread_counter_lock = threading.Lock()

ocr_objects = [ocr1, ocr2, ocr3]
ocr_parallel_count = len(ocr_objects)

ocr_semaphore = threading.Semaphore(value=ocr_parallel_cout)

with ocr_semaphore:
    current_thread_counter = 0
    try:
        parallel_thread_counter_lock.acquire()
        parallel_thread_counter += 1
        current_thread_counter = parallel_thread_counter % ocr_parallel_count
    finally:
        parallel_thread_counter_lock.release()

    selected_ocr = ocr_objects[current_thread_counter]
    ocr_dump  = selected_ocr.ocr(npArray)
