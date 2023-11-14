import numpy as np
import cv2


from lightlab.engine.onnx_predictor import OnnxPredictor


class Predictor(OnnxPredictor):
    pad_val = 114

    def __init__(
        self,
        model_path,
        imgsz=640,
        batch=1,
        device="cuda",
        warmup_epoch=1,
        conf=0.5,
        iou=0.5,
        **kwargs
    ) -> None:
        super().__init__(model_path, imgsz, batch, device, warmup_epoch, **kwargs)
        self.conf = conf
        self.iou = iou

    def postprocess(self, outputs: np.ndarray):
        outputs = outputs.transpose(0, 2, 1)
        results = []
        for output in outputs:
            # Get the number of rows in the outputs array
            rows = output.shape[0]

            # Lists to store the bounding boxes, scores, and class IDs of the detections
            boxes = []
            scores = []
            class_ids = []
            # Iterate over each row in the outputs array
            for i in range(rows):
                # Extract the class scores from the current row
                classes_scores = output[i][4:]

                # Find the maximum score among the class scores
                max_score = np.amax(classes_scores)

                # If the maximum score is above the confidence threshold
                if max_score < self.conf:
                    continue

                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = output[i][:4]

                # Calculate the scaled coordinates of the bounding box
                left = x - w / 2
                top = y - h / 2
                width = w
                height = h

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

            # Apply non-maximum suppression to filter out overlapping bounding boxes
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)

            # Iterate over the selected indices after non-maximum suppression
            if len(indices) > 0:
                results.append(
                    [
                        dict(box=boxes[i], score=scores[i], class_id=class_ids[i])
                        for i in indices
                    ]
                )
            else:
                results.append([])

        print(results)

    def preprocess(self, img: np.ndarray):
        im = super().preprocess(img) / 255.0
        return im


if __name__ == "__main__":
    model = Predictor("output/coco128/weights/best.onnx")
    file = r"datasets/coco128/val/images/000000000036.jpg"
    model(file)
