import logging

import cv2
from tflite_support.task import vision

logger = logging.getLogger(__name__)


class ImageEmbedder:
    def __init__(self):
        self.image_embedder = vision.ImageEmbedder.create_from_file(
            "/tmp/mobilenet_v3_embedder.tflite"
        )

    def get_embeddings(self, frame, detection):
        bgr_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_YUV2BGR_I420,
        )

        x1, y1, x2, y2 = detection[2]
        cropped_frame = bgr_frame[y1:y2, x1:x2]

        # cv2.imwrite(
        #     f"debug/frames/embeddings-{'{:.6f}'.format(time.time())}.jpg",
        #     cropped_frame,
        # )

        tensor_image = vision.TensorImage.create_from_array(cropped_frame)

        embedding = self.image_embedder.embed(tensor_image)

        # logger.debug(f"Embeddings for image: {embedding}")

        return embedding
