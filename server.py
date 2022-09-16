import datetime
import mimetypes
import os, json, cv2
import requests
import base64
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import io

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from findComponents import *
from shutil import copy
from data_processing.diagram import drawDiagram
from data_processing.calcVoltageAndCurrent import calcCurrentAndVoltage
from tqdm import tqdm
from mmdet.apis import init_detector
from data_processing.writeSPICE import toSPICE

pd.set_option("mode.chained_assignment", None)

circuit_component_data = []
body_pinmap = None
vol_pinmap = None
pinmap = None
pinmap_shape = None

V = 5

MODEL_RESISTORAREA_PATH = "model/resistor-area.model.pt"
MODEL_RESISTORBODY_PATH = "model/resistor-body.model.pt"
MODEL_LINEAREA_PATH = "model/line-area.model.pt"
MODEL_LINEENDAREA_PATH = "model/line-endpoint.model.pt"
MODEL_RESISTORPIN_PATH = "model/resistor-pin.model.h5"
MODEL_LINEPIN_PATH = "model/line-pin.model.h5"

app = Flask(__name__, static_folder="./static")
app.config["JSON_AS_ASCII"] = False
app.secret_key = "f#@&v08@#&*fnvn"
app.permanent_session_lifetime = datetime.timedelta(hours=4)

CORS(app, resources={r"/*": {"origins": "*"}})

FILE_IMAGE = None
models = {}


@app.route("/")
def main():
    """
    서버 동작 체크 여부
    """
    return "Hello, I'm on"


@app.route("/spice", methods=["GET"])
def spice():
    return send_file(
        "./circuit_spice.cir",
        mimetype="text/plain",
        download_name="circuitSpice.cir",
        as_attachment=True,
    )


@app.route("/pinmap", methods=["GET"])
def pinmap():
    """
    브레드보드에서의 Pin위치를 전달하면 입력된 이미지 기준으로 해당하는 Pin의 x, y 픽셀 좌표를 반환한다.
    Pinmap은 backend/static/data/pinmap.json의 파일을 기준으로 한다.
    Pinmap은 사용자가 선택한 4개의 꼭짓점으로 변환된 이미지에 mapping한다. 이미지따라 핀의 픽셀 위치가 다 다르기 때문이다.
    ex) A01 => 130, 245
    ex) V225 => 110, 3000
    """
    global body_pinmap, vol_pinmap
    search_map = None
    row = None
    col = None
    search_pin = request.args.get("pin")

    try:
        assert search_pin != None
        assert search_pin[0] in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "V"]

        search_map = pd.concat(
            [vol_pinmap.iloc[:, 0:4], body_pinmap, vol_pinmap.iloc[:, 4:8]], axis=1
        )

        if search_pin[0] == "V":
            assert int(search_pin[2:]) >= 1 and int(search_pin[2:]) <= 25
            row, col = int(search_pin[2:]) - 1, search_pin[:2]
        else:
            assert int(search_pin[1:]) >= 1 and int(search_pin[1:]) <= 30
            row, col = int(search_pin[1:]) - 1, search_pin[0]

        x, y = search_map.xs(row)[col]["x"], search_map.xs(row)[col]["y"]

        return jsonify({"coord": [x, y]})

    except AttributeError as e:
        return jsonify({"message": "회로 이미지를 먼저 업로드하세요"})

    except AssertionError:
        return jsonify({"message": "유효한 브레드보드 핀 범위가 아님. (V11(V101)~V425, A1(A01)~J25)"})


@app.route("/resistor", methods=["GET", "POST"])
def resistor():
    """
    To-do: 만약 저항값이 잘못 검측되었을 때 필요로 함
    분석이 완료된 회로 계층의 저항값을 반환받는다.
    Ex) 회로가 1 - 1 - 2(병렬 저항)로 구성됨
    response = {
        "state": "success",
        "data": [[{'name': 'R0', 'value': 100}], [{'name': 'R1', 'value': 100}], [{'name': 'R2', 'value': 100}, {'name': 'R3', 'value': 100}]]
    }
    GET: 서버 메모리에 올라간 회로 계층 데이터 반환
    POST: 전달된 데이터를 바탕으로 저항값 데이터 초기화
    """
    global circuit_component_data

    if request.method == "GET":
        return jsonify({"state": "success", "data": circuit_component_data})

    if request.method == "POST":
        resistor_value = request.get_json()

        for r in resistor_value:
            for row in circuit_component_data:
                for col in row:
                    if r["name"] == col["name"]:
                        col["value"] = int(r["value"])

        return jsonify({"state": "success"})


@app.route("/draw", methods=["GET"])
def draw():
    """
    회로 다이어그램 그림을 반환함
    """
    if request.method == "GET":
        global circuit_component_data
        try:
            image_bytes = drawDiagram(V, circuit_component_data)

            return send_file(
                io.BytesIO(image_bytes),
                mimetype="image/jpeg",
                download_name="circuitDiagram.jpeg",
            )
        except Exception:
            return jsonify({"message": "회로 이미지를 먼저 업로드하세요"})


@app.route("/image", methods=["POST"])
def image():
    """
    /upload에서 업로드한 이미지 데이터와 전압 데이터를 전달받는다.
    이후 /detect를 호출하여 전기소자 예측 모델과 전기소자 위치 검출 모델로 데이터를 반환받는다.

    """
    if request.method == "POST":
        try:
            global FILE_IMAGE, V
            PADDING = 200
            target_image = None
            points = None
            scale = None

            if request.files:
                circuitImage = request.files["circuitImage"].read()
                target_image = cv2.imdecode(
                    np.frombuffer(circuitImage, np.uint8), cv2.IMREAD_COLOR
                )

            # 테스트 데이터를 위한 분기
            else:
                raise Exception("회로 이미지를 전송하지 않음")

            if request.form["points"]:
                data = json.loads(request.form["points"])

                if data.get("points") and len(data["points"]) == 4:
                    points = data["points"]
                else:
                    raise Exception("이미지 필수 정보 (브레드보드 꼭짓점)이 전달되지 않음")

                if data.get("scale"):
                    scale = 6.75
                else:
                    raise Exception("이미지 필수 정보 (스케일)이 전달되지 않음")

                if data.get("voltage"):
                    V = int(data["voltage"])
                else:
                    raise Exception("이미지 필수 정보 (전압)이 전달되지 않음")

            # 전달받은 4개의 포인트는 스케일이 적용되어 있다.
            # 웹에서 포인트를 선택하는 영역은 화면의 크기에 따라 해당하는 점 위치가 다르기 때문에
            # 실제 이미지 크기에 맞게 스케일링을 한다.
            # 현재 scale은 6.75로 고정되어있다.

            pts = []
            for point in points:
                pts.append([int(point[0] * scale), int(point[1] * scale)])

            base_point, target_image = toPerspectiveImage(
                target_image, np.array(pts), PADDING
            )

            _, buffer = cv2.imencode(".jpg", target_image)
            transformedImg_base64 = base64.b64encode(buffer).decode()

            component = detect(
                pts=base_point,
                target_image=target_image,
                scale=scale,
            )

            print(component["components"])

            result_data = {
                "transformedImg": transformedImg_base64,
                "basePoint": base_point.tolist(),
                "voltage": V,
                "scale": 0.25,
                "components": component["components"],
            }

            return jsonify(result_data)
        except Exception as e:
            return jsonify({"message": str(e)})


@app.route("/calc", methods=["get"])
def calc():
    """
    분석된 회로에서의 이론적인 노드 전압과 출력 전류 그리고 합성저항값을 반환받는다.
    """
    global circuit_component_data

    if circuit_component_data is None:
        return jsonify({"message": "회로 사진을 먼저 업로드하세요"})

    if V is None:
        return jsonify({"message": "전압값이 지정되어있지 않음"})

    R_TH, I, NODE_VOL = calcCurrentAndVoltage(V, circuit_component_data)

    return jsonify(
        {
            "circuit_analysis": {
                "r_th": str(R_TH),
                "node_current": str(I),
                "node_voltage": str(NODE_VOL),
            }
        }
    )


@app.route("/network", methods=["GET", "POST"])
def network():
    """
    네트워크의 계층을 찾지않고 검출된 순서대로 정렬된 각 전기소자들을 회로가 결선된 순서에 맞게 레이어를 구성한다.
    """
    if request.method == "POST":
        global circuit_component_data
        components = request.get_json()

        lines = pd.DataFrame(components["Line"])
        resistors = pd.DataFrame(components["Resistor"])

        components = pd.concat([lines, resistors], axis=1).transpose()

        circuit = findNetwork(components)

        # 아래의 코드는 위에서 계층을 찾으면 그걸 토대로 저항소자만 빼오는 코드
        # 위 데이터의 결과는 [[{'name': 'R0', 'value': 100}], [{'name': 'R1', 'value': 100}], [{'name': 'R2', 'value': 100}, {'name': 'R3', 'value': 100}]]

        if isinstance(circuit, dict):
            return jsonify({"message": circuit["message"]})

        table = {}

        for i in range(len(circuit)):
            row = circuit.iloc[i]

            if "R" not in row["name"]:
                continue

            d = {"name": row["name"], "value": row["value"]}

            if table.get(row.layer) is None:
                table[int(row.layer)] = [d]
            else:
                table[int(row.layer)].append(d)

        table = [value for _, value in table.items()]

        circuit_component_data = table
        print(circuit)
        toSPICE(circuit, V, "./circuit_spice.cir")

        return jsonify({"network": table})

    if request.method == "GET":
        if circuit_component_data is not None:
            return jsonify({"network": circuit_component_data})
        else:
            return jsonify({"message": "회로 이미지를 먼저 업로드하세요"})


# @app.route("/detect", methods=["POST"])
def detect(pts: np.ndarray, target_image: np.ndarray, scale: float):
    """
    전달받은 이미지 데이터에서 전기소자를 예측하고 그것의 위치를 찾는다.
    """
    global circuit_component_data, vol_pinmap, body_pinmap, start_point, find_pincoords_resi_model, find_pincoords_line_model

    init()  # 검출을 위해 데이터 초기화 및 모델 로딩(로딩이 되어있지 않다면)

    canvas_image = target_image.copy()

    resistor_area_pd = detectArea(
        models["resistor.area"], "resistor-area", target_image, 0.5
    )
    resistor_body_pd = detectArea(
        models["resistor.body"], "resistor-body", target_image, 0.5
    )
    line_area_pd = detectArea(models["line.area"], "line-area", target_image, 0.5)
    line_endarea_pd = detectArea(
        models["line.endpoint"], "line-endpoint", target_image, 0.5
    )

    base_point = np.array(pts, np.float32)
    transform_mtrx = cv2.getPerspectiveTransform(start_point, base_point)

    initializePinmaps(
        body_pinmap, vol_pinmap, transform_mtrx
    )  # 왜곡된 이미지에 맞게 핀맵의 좌표도 똑같이 변환
    search_map = pd.concat(
        [vol_pinmap.iloc[:, 0:4], body_pinmap, vol_pinmap.iloc[:, 4:8]], axis=1
    )

    print("resistor_area_pd:", len(resistor_area_pd))
    print("resistor_body_pd:", len(resistor_body_pd))
    print("line_area_pd:", len(line_area_pd))
    print("line_endarea_pd:", len(line_endarea_pd))

    components = {"Resistor": [], "Unknown": []}

    detected_resistor_components = set_resistor_component(
        resistor_area_pd,
        resistor_body_pd,
        search_map,
        base_point,
        target_image,
        canvas_image,
        body_pinmap,
        vol_pinmap,
        models["resistor.pin"],
    )
    components["Resistor"] = detected_resistor_components

    # 전선영역이 검출되었다면
    if len(line_area_pd) > 0:
        table = line_contains_table(line_area_pd, line_endarea_pd)

        detected_line_components = set_line_component(
            line_area_pd,
            line_endarea_pd,
            table,
            search_map,
            base_point,
            target_image,
            canvas_image,
            body_pinmap,
            vol_pinmap,
            models["line.pin"],
        )
        components["Line"] = detected_line_components["Line"]
        components["Unknown"] = detected_line_components["Unknown"]

        for lineAreaIdx in table.values():
            r = int(random.random() * 255)
            g = int(random.random() * 255)
            b = int(random.random() * 255)

            linearea = line_area_pd.iloc[lineAreaIdx]

            lineareaMinPoint = round(linearea.xmin) - 30, round(linearea.ymin) - 30
            lineareaMaxPoint = round(linearea.xmax) + 30, round(linearea.ymax) + 30

            cv2.putText(
                canvas_image,
                f"linearea#{lineAreaIdx}",
                (lineareaMinPoint[0], lineareaMinPoint[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 255),
                2,
            )
            cv2.rectangle(
                canvas_image, lineareaMinPoint, lineareaMaxPoint, (b, g, r), 10
            )
    else:
        components["Line"] = {}

    return {"components": components}


def init():
    global body_pinmap, vol_pinmap, pinmap, pinmap_shape, start_point

    PADDING = 0

    pinmap = json.load(open("static/data/pinmap.json"))
    pinmap_shape = pinmap["shape"]

    body_pinmap = breadboard_bodypin_df(pinmap, PADDING)
    vol_pinmap = breadboard_voltagepin_df(pinmap, PADDING)

    start_point = np.array(
        [
            [PADDING, PADDING],
            [pinmap_shape[1] + PADDING, PADDING],
            [pinmap_shape[1] + PADDING, pinmap_shape[0] + PADDING],
            [PADDING, pinmap_shape[0] + PADDING],
        ],
        dtype=np.float32,
    )


def model_loading():
    global models
    print("Init...")

    keys = [
        "resistor.area",
        "resistor.body",
        "line.area",
        "line.endpoint",
        "resistor.pin",
        "line.pin",
    ]

    model_paths = [
        MODEL_RESISTORAREA_PATH,
        MODEL_RESISTORBODY_PATH,
        MODEL_LINEAREA_PATH,
        MODEL_LINEENDAREA_PATH,
        MODEL_RESISTORPIN_PATH,
        MODEL_LINEPIN_PATH,
    ]

    for key, model_path in zip(keys[:4], model_paths[:4]):
        models[key] = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)

    for key, model_path in zip(keys[4:], model_paths[4:]):
        models[key] = tf.keras.models.load_model(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="서버 시작")
    parser.add_argument("--debug", default=False)
    parser.add_argument("--port", default=7080)
    args = parser.parse_args()

    model_loading()
    app.run(debug=args.debug, host="0.0.0.0", port=args.port)
