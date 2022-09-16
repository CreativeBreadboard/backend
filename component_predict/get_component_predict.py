from model_pipeline import LoadDetectionModel
import pandas as pd
import cv2

CPM_PATH = "model/component_predict_model.onnx"
CPM_CONF_PATH = "model/component_predict_model_conf.py"
CPM_CLASSES = ("line-area", "line-endpoint", "resistor-area", "resistor-body")


def get_component(img):
    model = LoadDetectionModel(CPM_PATH, CPM_CONF_PATH)
    model.setInputImg(img)
    result, categories = model.predict()

    result_dataframe = pd.DataFrame(
        {}, columns=["xmin", "ymin", "xmax", "ymax", "confidence", "name"]
    )

    print(len(result[0]))

    for idx, (row, category) in enumerate(zip(result[0], categories)):
        if row[4] != 0.0:
            result_dataframe.loc[idx, "xmin"] = row[0]
            result_dataframe.loc[idx, "ymin"] = row[1]
            result_dataframe.loc[idx, "xmax"] = row[2]
            result_dataframe.loc[idx, "ymax"] = row[3]
            result_dataframe.loc[idx, "confidence"] = row[4]
            result_dataframe.loc[idx, "name"] = CPM_CLASSES[category]

    return result_dataframe
