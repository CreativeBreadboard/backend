from typing import *
from mmcv import Config
import onnxruntime
import torch
import numpy as np
import cv2

####################################################
# ONNX 형식 모델을 로드 및 전처리를 진행하고 결과를 예측한다.
####################################################


class LoadDetectionModel:
    input_arr: torch.Tensor = None
    input_img: np.ndarray = None
    input_img_path: str = None
    img_scale: tuple = None

    def __init__(
        self, model_path: str, model_cfg: str, input_img: str or np.ndarray = None
    ):
        self.model = onnxruntime.InferenceSession(model_path)
        self.model_cfg = model_cfg

        if input_img is not None:
            if isinstance(input_img, str):
                self.input_img_path = input_img

            elif isinstance(input_img, np.ndarray):
                self.input_img = input_img

    def data_pipeline(self) -> torch.Tensor:
        cfg = Config.fromfile(self.model_cfg)
        img = None
        transforms = None

        for pipeline in cfg.test_pipeline:
            if "img_scale" in pipeline:
                self.img_scale = pipeline["img_scale"]

            if "transforms" in pipeline:
                transforms = pipeline["transforms"]
                break

        assert transforms is not None, "Failed to find `transforms`"
        norm_config_li = [_ for _ in transforms if _["type"] == "Normalize"]
        assert len(norm_config_li) == 1, "`norm_config` should only have one"
        norm_config = norm_config_li[0]

        if self.input_img_path is not None:
            img = cv2.imread(self.input_img_path, cv2.IMREAD_COLOR)

        elif self.input_img is not None:
            img = self.input_img

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_shape = (1, 3, self.img_scale[1], self.img_scale[0])

        img = cv2.resize(img, input_shape[2:][::-1])
        mean = np.array(norm_config["mean"], dtype=np.float32)
        std = np.array(norm_config["std"], dtype=np.float32)

        normalized = (img - mean) / std
        img = normalized.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).float().requires_grad_(True)

        self.input_arr = img.cpu().detach().numpy()

    def setInputImg(self, input_img: np.ndarray):
        self.input_img = input_img

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.input_arr is None:
            self.data_pipeline()

        sess_run = self.model.run([], {"input": self.input_arr})
        results = sess_run[0]

        # 원본 이미지 스케일에 맞게 결과값 조정
        for rid, result in enumerate(results):
            for rowid, row in enumerate(result):
                results[rid][rowid][0] = row[0] * (
                    self.input_img.shape[1] / self.img_scale[0]
                )
                results[rid][rowid][1] = row[1] * (
                    self.input_img.shape[0] / self.img_scale[1]
                )
                results[rid][rowid][2] = row[2] * (
                    self.input_img.shape[1] / self.img_scale[0]
                )
                results[rid][rowid][3] = row[3] * (
                    self.input_img.shape[0] / self.img_scale[1]
                )

        categories = sess_run[1][0]

        return (results, categories)
