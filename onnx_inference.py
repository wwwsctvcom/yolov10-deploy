import cv2
import numpy as np
import onnxruntime
from pathlib import Path


class OnnxInference:

    def __init__(self, onnx_model: str, conf_threshold: float = 0.2):
        self.onnx_model = onnx_model

        if not Path(onnx_model).suffix == ".onnx":
            raise ValueError("model must be onnx file.")

        self.conf_threshold = conf_threshold

        # src image
        self.img_height = 0
        self.img_width = 0

        # input and output info
        self.input_names = []
        self.input_height = 640
        self.input_width = 640

        self.output_names = []

        # init onnx model session
        self.session = onnxruntime.InferenceSession(onnx_model, providers=onnxruntime.get_available_providers())

        # init model feed
        self.onnx_input_init()
        self.onnx_output_init()

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = self.preprocess(image)

        outputs = self.session.run(self.output_names, {self.input_names[0]: image})

        classes, boxes, confidences = self.postprocess(outputs[0])
        return classes, boxes, confidences

    def onnx_input_init(self):
        model_inputs = self.session.get_inputs()

        for node in self.session.get_inputs():
            self.input_names.append(node.name)

        input_shape = model_inputs[0].shape
        self.input_height = int(input_shape[2])
        self.input_width = int(input_shape[3])

    def onnx_output_init(self):
        for node in self.session.get_outputs():
            self.output_names.append(node.name)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        self.img_height, self.img_width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
        image = cv2.resize(image, (self.input_width, self.input_height))  # resize
        image = image / 255.0  # normalize
        image = image.transpose(2, 0, 1)  # hwc to chw
        image = image[np.newaxis, :, :, :].astype(np.float32)  # expand dim
        return image

    def postprocess(self, output) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param output: [max_object_num, 6], x_min, y_min, x_max, y_max
        :return: class id, boxes, confidence
        """
        output = output.squeeze()

        boxes = output[:, :-2]
        confidences = output[:, -2]
        classes = output[:, -1].astype(int)

        mask = confidences > self.conf_threshold
        boxes = boxes[mask, :]
        confidences = confidences[mask]
        classes = classes[mask]

        # rescale boxes
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return classes, boxes, confidences

    @staticmethod
    def draw_boxes(image, boxes, classes_ids, confidences, class_names):
        """
        Draws bounding boxes on the image along with class names and confidences.

        Parameters:
            image (np.array): The original image.
            boxes (np.array): An array of shape (N, 4) where N is the number of boxes and each row contains the coordinates of the top-left and bottom-right corners of each box.
            classes_ids (np.array): An array of class IDs corresponding to each box.
            confidences (np.array): An array of confidence scores for each detection.
            class_names (list): A dictionary mapping class IDs to their names.

        Returns:
            np.array: Image with drawn boxes and labels.
        """
        for i in range(len(boxes)):
            # Unpack the box coordinates
            x1, y1, x2, y2 = boxes[i]

            # Draw the rectangle
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)

            # Get the class name from ID
            class_name = class_names[classes_ids[i]]

            # Prepare the label text
            label = f"{class_name}: {confidences[i]:.2f}"

            # Put the label on the image
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image
