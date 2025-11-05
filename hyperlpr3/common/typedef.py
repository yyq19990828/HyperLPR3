import numpy as np

PLATE_TYPE_BLUE = 0
PLATE_TYPE_GREEN = 1
PLATE_TYPE_YELLOW = 2

INFER_ONNX_RUNTIME = 0
INFER_MNN = 1

DETECT_LEVEL_LOW = 0
DETECT_LEVEL_HIGH = 1

MONO = 0    # 单层车牌
DOUBLE = 1  # 双层车牌

UNKNOWN = -1                         # 未知车牌
BLUE = 0                             # 蓝牌
YELLOW_SINGLE = 1                    # 黄牌单层
WHILE_SINGLE = 2                     # 白牌单层
GREEN = 3                            # 绿牌新能源
BLACK_HK_MACAO = 4                   # 黑牌港澳
HK_SINGLE = 5                        # 香港单层
HK_DOUBLE = 6                        # 香港双层
MACAO_SINGLE = 7                     # 澳门单层
MACAO_DOUBLE = 8                     # 澳门双层
YELLOW_DOUBLE = 9                    # 黄牌双层


def code_filter(code: str) -> int:
    """Determines the license plate type based on the recognized plate code.

    This function analyzes the license plate text to identify the plate type
    by checking for specific patterns and characters that are unique to
    different plate categories (e.g., military, Hong Kong, Macao, new energy).

    Args:
        code (str): The recognized license plate text string.

    Returns:
        int: The identified plate type constant (e.g., BLUE, GREEN, YELLOW_SINGLE).
            Returns UNKNOWN if the plate type cannot be determined from the code.
    """
    plate_type = UNKNOWN
    if code[0] == 'W' and code[1] == 'J':
        plate_type = WHILE_SINGLE
    elif len(code) == 8:
        plate_type = GREEN
    elif '学' in code:
        plate_type = BLUE
    elif '港' in code:
        plate_type = BLACK_HK_MACAO
    elif '澳' in code:
        plate_type = BLACK_HK_MACAO
    elif '警' in code:
        plate_type = WHILE_SINGLE
    elif '粤Z' in code:
        plate_type = BLACK_HK_MACAO

    return plate_type


class Plate(object):
    """Represents a detected license plate with its properties.

    This class encapsulates all information about a detected license plate,
    including its position, recognized text, confidence scores, type, and
    layer configuration (single or double layer).

    Attributes:
        vertex (np.ndarray): Four corner points of the license plate, shape (4, 2).
        det_bound_box (np.ndarray): Detection bounding box coordinates.
        plate_code (str): Recognized license plate text.
        rec_confidence (float): Recognition confidence score.
        dex_bound_confidence (float): Detection bounding box confidence score.
        left_top (np.ndarray): Top-left corner point.
        right_top (np.ndarray): Top-right corner point.
        right_bottom (np.ndarray): Bottom-right corner point.
        left_bottom (np.ndarray): Bottom-left corner point.
        plate_type (int): Type of license plate (e.g., BLUE, YELLOW_SINGLE, GREEN).
        layer_num (int): Layer configuration - MONO (0) for single layer, DOUBLE (1) for double layer.
    """

    def __init__(self,
                 vertex: np.ndarray,
                 plate_code: str,
                 rec_confidence: float,
                 det_bound_box,
                 dex_bound_confidence: float,
                 plate_type: int,
                 layer_num: int = MONO):
        """Initializes a Plate object with detection and recognition results.

        Args:
            vertex (np.ndarray): Four corner points of the license plate, shape (4, 2).
            plate_code (str): Recognized license plate text.
            rec_confidence (float): Recognition confidence score.
            det_bound_box: Detection bounding box coordinates.
            dex_bound_confidence (float): Detection bounding box confidence score.
            plate_type (int): Type of license plate.
            layer_num (int, optional): Layer configuration, MONO (0) or DOUBLE (1).
                Defaults to MONO.
        """
        assert vertex.shape == (4, 2)
        self.vertex = vertex
        self.det_bound_box = det_bound_box
        self.plate_code = plate_code
        self.rec_confidence = rec_confidence
        self.dex_bound_confidence = dex_bound_confidence

        self.left_top, self.right_top, self.right_bottom, self.left_bottom = vertex
        self.plate_type = plate_type
        self.layer_num = layer_num

    def to_dict(self):
        """Converts the plate information to a dictionary.

        Returns:
            dict: Dictionary containing plate_code, rec_confidence, det_bound_box,
                plate_type, and layer_num.
        """
        return dict(plate_code=self.plate_code, rec_confidence=self.rec_confidence,
                    det_bound_box=self.det_bound_box, plate_type=self.plate_type,
                    layer_num=self.layer_num)

    def to_result(self):
        """Converts the plate information to a compact result list.

        Returns:
            list: List containing [plate_code, rec_confidence, plate_type,
                det_bound_box, layer_num].
        """
        return [self.plate_code, self.rec_confidence, self.plate_type,
                self.det_bound_box.tolist(), self.layer_num]

    def to_full_result(self):
        """Converts the plate information to a full result list with vertex points.

        Returns:
            list: List containing [plate_code, rec_confidence, plate_type,
                det_bound_box, vertex, layer_num].
        """
        return [self.plate_code, self.rec_confidence, self.plate_type,
                self.det_bound_box.tolist(), self.vertex.tolist(), self.layer_num]

    def __dict__(self):
        return self.to_dict()

    def __str__(self):
        return str(self.to_dict())
