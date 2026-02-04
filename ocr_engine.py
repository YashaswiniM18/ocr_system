from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=False,
    enable_mkldnn=False
)

def run_ocr(image):
    result = ocr.ocr(image, cls=True)
    texts = [line[1][0] for line in result[0]]
    return texts


def run_ocr(image):
    result = ocr.ocr(image)
    texts = []

    if result:
        for line in result:
            for word in line:
                texts.append(word[1][0])

    return texts





