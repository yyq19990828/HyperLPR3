from .config.settings import onnx_runtime_config as ort_cfg
from .inference.pipeline import LPRMultiTaskPipeline
from .common.typedef import *
from os.path import join
from .config.settings import _DEFAULT_FOLDER_
from .config.configuration import initialization


initialization()

class LicensePlateCatcher(object):
    """High-level API for Chinese license plate recognition.

    This class provides a simple interface for detecting and recognizing
    Chinese license plates from images. It supports both single-layer and
    double-layer plates, and can classify plates into various types
    (blue, yellow, green, etc.).

    The catcher automatically handles the complete pipeline including:
    - License plate detection
    - Single/double layer identification
    - Text recognition (with special handling for double-layer plates)
    - Plate type classification

    Attributes:
        pipeline (LPRMultiTaskPipeline): The underlying recognition pipeline.

    Example:
        >>> catcher = LicensePlateCatcher(detect_level=DETECT_LEVEL_LOW)
        >>> results = catcher(image)
        >>> for result in results:
        >>>     plate_code, confidence, plate_type, bbox, layer = result
        >>>     print(f"Plate: {plate_code}, Layer: {layer}")
    """

    def __init__(self,
                 inference: int = INFER_ONNX_RUNTIME,
                 folder: str = _DEFAULT_FOLDER_,
                 detect_level: int = DETECT_LEVEL_LOW,
                 logger_level: int = 3,
                 full_result: bool = False):
        """Initializes the LicensePlateCatcher with specified configuration.

        Args:
            inference (int, optional): Inference engine type. Currently only
                INFER_ONNX_RUNTIME is supported. Defaults to INFER_ONNX_RUNTIME.
            folder (str, optional): Directory containing model files. Defaults
                to the package's default model directory.
            detect_level (int, optional): Detection level controlling accuracy
                vs speed tradeoff. Options are:
                - DETECT_LEVEL_LOW: Fast detection with 320x320 input (default)
                - DETECT_LEVEL_HIGH: More accurate detection with 640x640 input
            logger_level (int, optional): ONNX Runtime logging level (0-3).
                Higher values mean less verbose logging. Defaults to 3.
            full_result (bool, optional): If True, results include vertex points
                for each detected plate. Defaults to False.

        Raises:
            NotImplemented: If unsupported inference engine or detect_level is specified.
        """
        if inference == INFER_ONNX_RUNTIME:
            from hyperlpr3.inference.multitask_detect import MultiTaskDetectorORT
            from hyperlpr3.inference.recognition import PPRCNNRecognitionORT
            from hyperlpr3.inference.classification import ClassificationORT
            import onnxruntime as ort
            ort.set_default_logger_severity(logger_level)

            if detect_level == DETECT_LEVEL_LOW:
                # print(join(folder, ort_cfg['det_model_path_320x']))
                det = MultiTaskDetectorORT(join(folder, ort_cfg['det_model_path_320x']), input_size=(320, 320))
            elif detect_level == DETECT_LEVEL_HIGH:
                det = MultiTaskDetectorORT(join(folder, ort_cfg['det_model_path_640x']), input_size=(640, 640))
            else:
                raise NotImplemented
            rec = PPRCNNRecognitionORT(join(folder, ort_cfg['rec_model_path']), input_size=(48, 160))
            cls = ClassificationORT(join(folder, ort_cfg['cls_model_path']), input_size=(96, 96))
            self.pipeline = LPRMultiTaskPipeline(detector=det, recognizer=rec, classifier=cls, full_result=full_result)
        else:
            raise NotImplemented

    def __call__(self, image: np.ndarray, *args, **kwargs):
        """Detects and recognizes license plates in an image.

        This method performs end-to-end license plate recognition, including
        detection, layer identification (single/double), text recognition,
        and plate type classification.

        Args:
            image (np.ndarray): Input image in BGR format with shape (H, W, 3).
            *args: Variable length argument list (unused).
            **kwargs: Arbitrary keyword arguments (unused).

        Returns:
            list: List of recognition results. Each result format depends on
                the full_result parameter:

                If full_result=False (default):
                    [plate_code, rec_confidence, plate_type, det_bound_box, layer_num]
                    - plate_code (str): Recognized license plate text
                    - rec_confidence (float): Recognition confidence score (0-1)
                    - plate_type (int): Plate type constant (e.g., BLUE, YELLOW_SINGLE)
                    - det_bound_box (list): Detection box [x1, y1, x2, y2]
                    - layer_num (int): MONO (0) for single layer, DOUBLE (1) for double layer

                If full_result=True:
                    [plate_code, rec_confidence, plate_type, det_bound_box, vertex, layer_num]
                    - vertex (list): Four corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        return self.pipeline(image)
