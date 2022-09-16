import random
import cv2
import numpy as np
import torch
import pandas as pd
import math
from resistance_value import get_resistance_value

BREADBOARD_COL_INDEX = [
    "A",
    "A",
    "B",
    "B",
    "C",
    "C",
    "D",
    "D",
    "E",
    "E",
    "F",
    "F",
    "G",
    "G",
    "H",
    "H",
    "I",
    "I",
    "J",
    "J",
]
BREADBOARD_ROW_INDEX = range(30)

PADDING = 0


def toPerspectiveImage(img, points, padding=0):
    """
    검출할 img를 points 기준으로 padding을 설정하여 원근변환을 한다.
    """
    if points.ndim != 2:
        points = points.reshape((-1, 2))

    sm = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    topLeft = points[np.argmin(sm)]
    bottomRight = points[np.argmax(sm)]
    topRight = points[np.argmin(diff)]
    bottomLeft = points[np.argmax(diff)]

    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])

    width = max([w1, w2])
    height = max([h1, h2])

    pts2 = np.float32(
        [
            [padding, padding],
            [width - 1 + padding, padding],
            [width - 1 + padding, height - 1 + padding],
            [padding, height - 1 + padding],
        ]
    )

    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    return pts2, cv2.warpPerspective(
        img, mtrx, (width + 2 * padding, height + 2 * padding), flags=cv2.INTER_CUBIC
    )


def area_padding(
    old_area,
    from_: tuple,
    to_: tuple,
    canvas_start: tuple or list,
    canvas_to: tuple or list,
    expand_to=0,
    blank=False,
):
    """
    타겟 영역에서 직사각형 영역 from_에서 to_ 까지 crop한다.
    padding이 이뤄진 전체 영역인 canvas_start, canvas_to를 가지고
    새롭게 crop 하는영역이 관심영역 (브레드보드 영역 안쪽)에만 잘리게 한다.

    expand_to로 중점을 중심으로 주위 핀을 찾기위해 확장한다.
    """

    # 범위를 넘어가나?
    x_ = [from_[0], to_[0]]
    y_ = [from_[1], to_[1]]

    if from_[0] > to_[0]:  # 오른쪽으로 범위가 넘어감
        # y_ = [canvas_start[1], canvas_to[1]]
        x_ = [canvas_start[0], to_[0]]

    if to_[0] < from_[0]:  # 왼쪽으로 범위가 넘어감
        # y_ = canvas_start[1], canvas_to[3]
        x_ = [from_[0], canvas_to[0]]

    if to_[1] < from_[1]:  # 위쪽으로 범위가 넘어감
        y_ = [from_[1], canvas_to[1]]
        # x_ = canvas_start[0], canvas_to[0]

    if from_[1] > to_[1]:  # 아래쪽으로 범위가 넘어감
        y_ = [canvas_start[1], to_[1]]
        # x_ = canvas_start[0], canvas_to[0]

    to_area = old_area[y_[0] : y_[1], x_[0] : x_[1]]

    c_x = to_area.shape[1] / 2
    c_y = to_area.shape[0] / 2

    # 110, 154 -> 350, 350
    # (350 - 110)/2 = 120, (350-154)/2 = 98

    if expand_to != 0:
        add_width = int((expand_to / 2 - c_x))
        add_height = int((expand_to / 2 - c_y))

        x_[0] -= add_width
        y_[0] -= add_height
        x_[1] += add_width
        y_[1] += add_height

        """
            지금 이 부분에서 문제가 있는 듯함 -> 일부 사진에서 포인트가 안맞음
        """
        # padding으로 확대했는데 범위를 넘어가면..
        # 범위를 넘어간 만큼 가능한 공간에서 다시 재확장된다.
        a = 0
        b = 0
        c = 0
        d = 0

        if x_[1] > canvas_to[0]:
            x_[1] -= add_width
            x_[0] -= add_width
            a -= add_width
            b += add_width
            # print(" xmax bound")

        if x_[0] < canvas_start[0]:
            x_[0] += add_width
            x_[1] += add_width
            a += add_width
            b -= add_width
            # print(" xmin bound")

        if y_[1] > canvas_to[1]:
            y_[1] -= add_height
            y_[0] -= add_height
            c -= add_height
            d += add_height
            # print(" ymax bound")

        if y_[0] < canvas_start[1]:
            y_[1] += add_height
            y_[0] += add_height
            c += add_height
            d -= add_height
            # print(" ymin bound")

        if blank:
            canvas = cv2.copyMakeBorder(
                to_area,
                add_height - c,
                add_height - d,
                add_width - a,
                add_width - b,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            # print(add_height-c, add_height-d, add_width-a, add_width-b)

            return (x_[0], y_[0]), (x_[1], y_[1]), canvas
        else:
            # padding 만큼 확장된 결과
            expanded = old_area[y_[0] : y_[1], x_[0] : x_[1]]
            return (x_[0], y_[0]), (x_[1], y_[1]), expanded
    else:
        return (x_[0], y_[0]), (x_[1], y_[1]), to_area


def rectArea(df):
    """
    영역 면적을 구하는 함수
    """
    return (df.xmax - df.xmin) * (df.ymax - df.ymin)


def processDataFrame(origin_data, column_name, confidence=0.1):
    """
    confidence를 기준으로 예측된 컴포넌트의 영역, 중심점, 길이, 넓이, 원점에서 떨어진 거리등을 데이터프레임에 추가한다.
    """
    df = origin_data[
        (origin_data["name"] == column_name) & (origin_data["confidence"] > confidence)
    ].copy()

    if len(df) > 0:
        df["area"] = df.apply(rectArea, axis=1)
        df["center_x"] = df.apply(lambda row: int((row.xmax + row.xmin) / 2), axis=1)
        df["center_y"] = df.apply(lambda row: int((row.ymax + row.ymin) / 2), axis=1)
        df["length"] = df.apply(lambda row: int((row.xmax - row.xmin)), axis=1)
        df["width"] = df.apply(lambda row: int((row.ymax - row.ymin)), axis=1)
        df["distance_from_origin"] = df.apply(
            lambda row: int((row.xmin + row.ymin)), axis=1
        )
        df = df.sort_values(by=["distance_from_origin"], ascending=True)

        return df
    else:
        return pd.DataFrame({})


def detectArea(model, name: str, target: np.ndarray, confidence: float):
    detected_area = pd.DataFrame(model(target).pandas().xyxy[0])
    print("detected_area: ", len(detected_area))

    if len(detected_area) > 0:
        detected_area = processDataFrame(detected_area, name, confidence)
    else:
        detected_area = {}

    return detected_area


def findCandidateCoords(area_start, area_end, bodymap, volmap):
    """
    검출된 컴포넌트 주위로 핀이 꽂혀있을 영역을 찾음
    어떤 컴포넌트가 C15, C18에 꽂혀있다면 그 주위의 D15~D18, B15~B18 등을 다 같이 반환함
    """
    global PADDING
    search_map = None
    table_idx = []

    index_map = {
        0: "V1",
        1: "V2",
        2: "A",
        3: "B",
        4: "C",
        5: "D",
        6: "E",
        7: "F",
        8: "G",
        9: "H",
        10: "I",
        11: "J",
        12: "V3",
        13: "V4",
    }

    search_map = pd.concat([volmap.iloc[:, 0:4], bodymap, volmap.iloc[:, 4:8]], axis=1)

    pin_x = [
        search_map.iloc[:, col].mean() - PADDING
        for col in range(0, len(search_map.columns), 2)
    ]
    pin_x = np.array(pin_x, np.uint32)
    range_x = np.array(np.where(((pin_x >= area_start[0]) & (pin_x <= area_end[0]))))[
        0
    ].tolist()
    col_name = np.array([index_map[idx] for idx in range_x], np.str0)

    isVContains = np.array(["V" in col for col in col_name], np.bool_)

    vol_map = col_name[isVContains == True]
    pin_map = col_name[isVContains == False]

    for col in pin_map:
        pin_y = [
            search_map.iloc[row, range(5, 24, 2)].mean() - PADDING for row in range(30)
        ]

        pin_y = np.array(pin_y, np.float64)
        range_y = np.array(
            np.where(((pin_y >= area_start[1]) & (pin_y <= area_end[1])))
        )[0].tolist()
        row_name = [idx + 1 for idx in range_y]

        for row in row_name:
            table_idx.append(f"{col}{row}")

    for col in vol_map:
        pin_y = [
            search_map.iloc[row, [1, 3, 25, 27]].mean() - PADDING for row in range(25)
        ]

        pin_y = np.array(pin_y, np.float64)
        range_y = np.array(
            np.where(((pin_y >= area_start[1]) & (pin_y <= area_end[1])))
        )[0].tolist()
        row_name = [idx + 1 for idx in range_y]

        for row in row_name:
            table_idx.append(f"{col}{row}")

    return table_idx


def xyToHomocoords(x, y):
    """
    [x, y] => [[x], [y], [1]] 형식의 좌표로 변환
    """
    return np.array([[x], [y], [1]])


def homocoordsToxy(v):
    """
    [[x], [y], [w]] => [x/w, y/w] 형식의 좌표로 변환
    """
    return int(v[0][0] / v[0][2]), int(v[0][1] / v[0][2])


def imgNormalizing(src, scale_to):
    """
    이미지 데이터 표준화
    히스토그램 평형과 이미지 크기 조절 그리고 정규 분포를 거침
    """
    print(src.shape)
    img_yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    img = cv2.resize(img_output, (scale_to, scale_to))
    img = img.reshape(-1, scale_to, scale_to, 3)

    mean = np.mean(img, axis=(0, 1, 2, 3))
    std = np.std(img, axis=(0, 1, 2, 3))

    img = (img - mean) / (std + 1e-5)
    return img


def getXYPinCoords(model, src):
    """
    이미지에서 해당 소자의 pin-point를 검출함
    """
    if model:
        c = list(model.predict(src)[0])
        return c
    else:
        print("nomodel")
        return [0, 0]


def getPinCoords(search_map, candidates, coord, area_start):
    """
    일전에 이미지에서 예측한 x, y는 픽셀좌표이다.
    브레드보드의 핀좌표는 A10, A11... 과 같이 불연속점으로 이미지 좌표 x, y로 부터 pin좌표로 변환한다.
    예측된 x, y에서 가장 가까운 pin좌표를 찾아 그 값을 반환한다.
    """
    global PADDING

    distance = 1000000
    final_coord = [0, 0]
    pin_name = -1

    for candidate in candidates:
        if "V" in candidate:
            col = str(candidate[:2])
            row = int(candidate[2:]) - 1
        else:
            col = str(candidate[:1])
            row = int(candidate[1:]) - 1

        c_x = search_map.xs(row)[col]["x"] - area_start[0] - PADDING
        c_y = search_map.xs(row)[col]["y"] - area_start[1] - PADDING

        n_x = coord[0] - area_start[0]
        n_y = coord[1] - area_start[1]

        d = math.sqrt((c_x - n_x) ** 2 + (c_y - n_y) ** 2)

        if d < distance:
            distance = d
            final_coord = [c_x, c_y]
            pin_name = f"{col}{row+1}"

    # print(final_coord)

    return int(final_coord[0]), int(final_coord[1]), pin_name


def translate(point, scale_to, expand_to, area_start):
    """
    검출 좌표는 300, 300에서 찾아지므로
    expand_to로 스케일 변환을 하여 조정할 필요가 있다.
    예측된 좌표를 homocoords로 변환하고
    스케일링 후 area_start 만큼 이동한다.

    """
    point = xyToHomocoords(point[0], point[1])

    to_x_sf = expand_to / scale_to
    to_y_sf = expand_to / scale_to

    move = np.array(
        [
            [1, 0, area_start[0]],
            [0, 1, area_start[1]],
            [0, 0, 1],
        ]
    )

    scaling = np.array([[to_x_sf, 0, 0], [0, to_y_sf, 0], [0, 0, 1]])

    point = move @ scaling @ point

    return homocoordsToxy(point.T)


def initializePinmaps(body_pinmap, vol_pinmap, transform_mtrx):
    """
    pinmap을 초기화한다.
    이것은 이미지마다 크기와 비율이 다르다. 기준이 되는 Pinmap과는 형태가 사뭇다르기 때문에
    이에 맞게 x, y 픽셀좌표를 변환하기 위함이다
    """
    global PADDING

    for C in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        for R in range(30):
            x, y = transform_pts(body_pinmap, transform_mtrx, C, R)
            body_pinmap.xs(R)[C]["x"] = round(x) + PADDING
            body_pinmap.xs(R)[C]["y"] = round(y) + PADDING

            # cv2.circle(result, (body_pinmap.xs(R)[C]['x'] - PADDING, body_pinmap.xs(R)[C]['y'] - PADDING), 10, (0, 255, 0), cv2.FILLED)

    for V in ["V1", "V2", "V3", "V4"]:
        for R in range(25):
            x, y = transform_pts(vol_pinmap, transform_mtrx, V, R)
            vol_pinmap.xs(R)[V]["x"] = round(x) + PADDING
            vol_pinmap.xs(R)[V]["y"] = round(y) + PADDING

            # cv2.circle(result, (vol_pinmap.xs(R)[V]['x'] - PADDING, vol_pinmap.xs(R)[V]['y'] - PADDING), 10, (0, 255, 255), cv2.FILLED)


def transform_pts(df, mtrx, col, row):
    """
    pinmap dataframe에 저장된 x, y 픽셀 좌표를
    새로운 이미지에 맞게 변환하는 과정이다.
    """
    fixed_x = df.xs(row)[col]["x"]
    fixed_y = df.xs(row)[col]["y"]

    n_point = [[fixed_x], [fixed_y], [1]]

    new_point = mtrx @ n_point

    new_x = new_point[0] / new_point[2]
    new_y = new_point[1] / new_point[2]

    return new_x[0], new_y[0]


def breadboard_bodypin_df(pinmap, padding=0):
    """
    static/data/pinmap.json에 저장된 핀좌표 x, y를 브레드보드에서의 좌표로 매핑한다.
    (A~J)메인 영역에 대한 핀매핑(같은 행이 같은 노드, 같은 전위차를 갖음)
    """
    A = np.array(BREADBOARD_COL_INDEX)
    B = np.array(["x", "y"] * 10)

    pinmap_pd = pd.DataFrame(columns=[A, B], index=BREADBOARD_ROW_INDEX)

    p2 = np.uint32(pinmap["2"]["points"]).reshape(5, 30, 2)
    p2[:, :, 0] += pinmap["2"]["start"] + padding
    p2[:, :, 1] += padding

    p3 = np.uint32(pinmap["3"]["points"]).reshape(5, 30, 2)
    p3[:, :, 0] += pinmap["3"]["start"] + padding
    p3[:, :, 1] += padding

    pinmap_pd["A"] = p2[0, :]
    pinmap_pd["B"] = p2[1, :]
    pinmap_pd["C"] = p2[2, :]
    pinmap_pd["D"] = p2[3, :]
    pinmap_pd["E"] = p2[4, :]

    pinmap_pd["F"] = p3[0, :]
    pinmap_pd["G"] = p3[1, :]
    pinmap_pd["H"] = p3[2, :]
    pinmap_pd["I"] = p3[3, :]
    pinmap_pd["J"] = p3[4, :]

    return pinmap_pd


def breadboard_voltagepin_df(pinmap, padding=0):
    """
    static/data/pinmap.json에 저장된 핀좌표 x, y를 브레드보드에서의 좌표로 매핑한다.
    전압 영역에 대한 핀매핑 (같은 열이 같은 노드, 같은 전위차를 갖음)
    """
    A = np.array(["V1", "V1", "V2", "V2", "V3", "V3", "V4", "V4"])
    B = np.array(["x", "y"] * 4)

    v1 = np.uint32(pinmap["1"]["points"]).reshape(2, 25, 2)
    v1[:, :, 0] += pinmap["1"]["start"] + padding
    v1[:, :, 1] += padding

    v2 = np.uint32(pinmap["4"]["points"]).reshape(2, 25, 2)
    v2[:, :, 0] += pinmap["4"]["start"] + padding
    v2[:, :, 1] += padding

    pinmap_pd = pd.DataFrame(columns=[A, B], index=range(25))

    pinmap_pd["V1"] = v1[0, :]
    pinmap_pd["V2"] = v1[1, :]

    pinmap_pd["V3"] = v2[0, :]
    pinmap_pd["V4"] = v2[1, :]

    return pinmap_pd


def set_line_component(
    line_area_pd,
    line_endarea_pd,
    table,
    search_map,
    base_point,
    target_image,
    canvas_image,
    body_pinmap,
    vol_pinmap,
    find_pincoords_line_model,
):
    """
    딕셔너리 형태의 전선 컴포넌트를 담음

    line_area_pd: 검출된 전선 컴포넌트 정보가 담김
    line_endarea_pd: 검출된 전선끝점 컴포넌트 정보가 담김
    search_map: 브레드보드의 전체에 대한 x, y 픽셀좌표와 핀좌표 정보가 담김
    base_point: 브레드보드 꼭짓점 영역 좌표 [0, 0], [width, 0], [0, height], [width, height]
    target_image: 검출할 이미지
    vol_pinmap, body_pinmap: 전압 핀맵과 메인 영역 핀맵 데이터 프레임 (사실 search_map으로 해결 가능... 일전에 짠 코드라 구현하기 편한 상태로 냅둔듯)
    find_pincoords_line_model: 전기소자 전선 핀 검출기 모델
    """
    temp = {"Line": {}, "Unknown": []}

    for endAreaIdx in table.keys():
        endarea = line_endarea_pd.iloc[endAreaIdx]
        expand_to = 350

        endAreaminPoint = [round(endarea.xmin) - 15, round(endarea.ymin) - 15]
        endAreamaxPoint = [round(endarea.xmax) + 15, round(endarea.ymax) + 15]

        # 시작점이 <0이 나올 수가 있음..
        if endAreaminPoint[0] < 0:
            endAreaminPoint[0] = 0

        if endAreaminPoint[1] < 0:
            endAreaminPoint[1] = 0

        cv2.putText(
            canvas_image,
            f"endarea#{endAreaIdx}",
            (endAreaminPoint[0], endAreaminPoint[1] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 255),
            2,
        )

        area_start, area_end, area = area_padding(
            target_image,
            endAreaminPoint,
            endAreamaxPoint,
            base_point[0],
            base_point[2],
            0,
            False,
        )

        table_idx = findCandidateCoords(area_start, area_end, body_pinmap, vol_pinmap)
        normalized = imgNormalizing(area, scale_to=227)

        coords = getXYPinCoords(find_pincoords_line_model, normalized)

        pt1 = round(coords[0]), round(coords[1])
        pt1 = translate(pt1, 227, expand_to, area_start)
        x1, y1, pin1 = getPinCoords(search_map, table_idx, pt1, area_start)

        toLineArea = table[endAreaIdx]
        linearea = line_area_pd.iloc[toLineArea]

        if temp["Line"].get(f"L{toLineArea}") is not None:
            if temp["Line"][f"L{toLineArea}"].get("start"):
                temp["Line"][f"L{toLineArea}"]["end"] = pin1
                temp["Line"][f"L{toLineArea}"]["end_coord"] = [
                    x1 + endAreaminPoint[0],
                    y1 + endAreaminPoint[1],
                ]
                temp["Line"][f"L{toLineArea}"]["end_endAreaStart"] = endAreaminPoint
                temp["Line"][f"L{toLineArea}"]["end_endAreaEnd"] = endAreamaxPoint

            else:
                temp["Line"][f"L{toLineArea}"]["start"] = pin1
                temp["Line"][f"L{toLineArea}"]["start_coord"] = [
                    x1 + endAreaminPoint[0],
                    y1 + endAreaminPoint[1],
                ]
                temp["Line"][f"L{toLineArea}"]["start_endAreaStart"] = endAreaminPoint
                temp["Line"][f"L{toLineArea}"]["start_endAreaEnd"] = endAreamaxPoint

        elif temp["Line"].get(f"linearea#{toLineArea}") is None:
            line_component = {
                "class": "Line",
                "name": f"L{toLineArea}",
                "areaStart": [round(linearea.xmin), round(linearea.ymin)],
                "areaEnd": [round(linearea.xmax), round(linearea.ymax)],
            }

            if "V1" in pin1 or "V3" in pin1:
                line_component["end"] = pin1
                line_component["end_coord"] = [
                    x1 + endAreaminPoint[0],
                    y1 + endAreaminPoint[1],
                ]
                line_component["end_endAreaStart"] = endAreaminPoint
                line_component["end_endAreaEnd"] = endAreamaxPoint

            elif "V2" in pin1 or "V4" in pin1:
                line_component["start"] = pin1
                line_component["start_coord"] = [
                    x1 + endAreaminPoint[0],
                    y1 + endAreaminPoint[1],
                ]
                line_component["start_endAreaStart"] = endAreaminPoint
                line_component["start_endAreaEnd"] = endAreamaxPoint

            else:
                line_component["start"] = pin1
                line_component["start_coord"] = [
                    x1 + endAreaminPoint[0],
                    y1 + endAreaminPoint[1],
                ]
                line_component["start_endAreaStart"] = endAreaminPoint
                line_component["start_endAreaEnd"] = endAreamaxPoint

            temp["Line"][f"L{toLineArea}"] = line_component

        else:
            temp["Unknown"].append(
                {
                    "pin": pin1,
                    "coord": [x1 + endAreaminPoint[0], y1 + endAreaminPoint[1]],
                    "areaStart": endAreaminPoint,
                    "areaEnd": endAreamaxPoint,
                }
            )

        cv2.putText(
            canvas_image,
            pin1,
            (x1 + area_start[0], y1 + area_start[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 255),
            2,
        )
        cv2.circle(
            canvas_image,
            (x1 + area_start[0], y1 + area_start[1]),
            20,
            (255, 0, 255),
            cv2.FILLED,
        )

    return temp


def set_resistor_component(
    resistor_area_pd,
    resistor_body_pd,
    search_map,
    base_point,
    target_image,
    canvas_image,
    body_pinmap,
    vol_pinmap,
    find_pincoords_resi_model,
):
    """
    딕셔너리 형태의 저항 컴포넌트 정보를 담음

    resistor_area_pd: 검출된 저항 컴포넌트 정보가 담김
    search_map: 브레드보드의 전체에 대한 x, y 픽셀좌표와 핀좌표 정보가 담김
    base_point: 브레드보드 꼭짓점 영역 좌표 [0, 0], [width, 0], [0, height], [width, height]
    target_image: 검출할 이미지
    vol_pinmap, body_pinmap: 전압 핀맵과 메인 영역 핀맵 데이터 프레임 (사실 search_map으로 해결 가능... 일전에 짠 코드라 구현하기 편한 상태로 냅둔듯)
    find_pincoords_resi_model: 전기소자 저항 핀 검출기 모델
    """
    temp = {}
    r = int(random.random() * 255)
    g = int(random.random() * 255)
    b = int(random.random() * 255)

    temp_body_pds: pd.DataFrame = resistor_body_pd.loc[
        :, ["xmin", "xmax", "ymin", "ymax"]
    ]
    temp_area_pds = resistor_area_pd.loc[:, ["xmin", "xmax", "ymin", "ymax"]]

    for i, (_, res_body) in zip(range(len(resistor_area_pd)), temp_body_pds.iterrows()):
        data = resistor_area_pd.iloc[i]

        res_value = get_resistance_value(
            target_image[
                int(res_body.ymin) + 10 : int(res_body.ymax) - 10,
                int(res_body.xmin) + 10 : int(res_body.xmax) - 10,
            ],
        )

        if len(data) == 0:
            continue

        minPoint = round(data.xmin) - 15, round(data.ymin) - 15
        maxPoint = round(data.xmax) + 15, round(data.ymax) + 15

        expand_to = max([maxPoint[0] - minPoint[0], maxPoint[1] - minPoint[1]])
        area_start, area_end, area = area_padding(
            target_image,
            minPoint,
            maxPoint,
            base_point[0],
            base_point[2],
            expand_to,
            True,
        )

        cv2.rectangle(canvas_image, minPoint, maxPoint, (b, g, r), 10)

        table_idx = findCandidateCoords(area_start, area_end, body_pinmap, vol_pinmap)

        normalized = imgNormalizing(area, scale_to=300)
        coords = getXYPinCoords(find_pincoords_resi_model, normalized)

        pt1 = round(coords[0]), round(coords[1])
        pt2 = round(coords[2]), round(coords[3])

        pt1 = translate(pt1, 300, expand_to, area_start)
        pt2 = translate(pt2, 300, expand_to, area_start)

        x1, y1, pin1 = getPinCoords(search_map, table_idx, pt1, area_start)
        x2, y2, pin2 = getPinCoords(search_map, table_idx, pt2, area_start)

        temp[f"R{i}"] = {
            "class": "Resistor",
            "name": f"R{i}",
            "start": pin1,
            "end": pin2,
            "start_coord": [x1 + area_start[0], y1 + area_start[1]],
            "end_coord": [x2 + area_start[0], y2 + area_start[1]],
            "value": res_value,
            "areaStart": minPoint,
            "areaEnd": maxPoint,
        }

        # 이미지에 표시
        cv2.putText(
            canvas_image,
            pin1,
            (x1 + area_start[0], y1 + area_start[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
            2,
        )
        cv2.circle(
            canvas_image,
            (x1 + area_start[0], y1 + area_start[1]),
            20,
            (0, 0, 255),
            cv2.FILLED,
        )
        cv2.putText(
            canvas_image,
            pin2,
            (x2 + area_start[0], y2 + area_start[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
            2,
        )
        cv2.circle(
            canvas_image,
            (x2 + area_start[0], y2 + area_start[1]),
            20,
            (20, 0, 255),
            cv2.FILLED,
        )
        # 끝

    return temp


def line_contains_table(line_area, line_endarea):
    """
    전선영역과 전선꼭지영역을 개별적으로 찾기 때문에 전선꼭지영역이 어떤 전선영역에 포함되는지 관계를 찾아준다.
    """
    key_table = dict()

    print("line_area", len(line_area))
    print("line_endarea", len(line_endarea))

    for i in range(len(line_area)):
        area = line_area.iloc[i]

        for j in range(len(line_endarea)):
            endarea = line_endarea.iloc[j]

            # print(f"전선영역{i}와 전선연결부{j}와 비교 중")

            # print(f"{area.xmin} < {endarea.center_x} < {area.xmax} || {area.ymin} < {endarea.center_y} < {area.ymax}")

            # linearea 안에 포함된 lineend를 찾는다.
            if ((area.xmin < endarea.center_x) and (endarea.center_x < area.xmax)) and (
                (area.ymin < endarea.center_y) and (endarea.center_y < area.ymax)
            ):
                if key_table.get(j) != None:
                    # compareKey(key_table, i, key_table[j], line_area, endarea) # >> 키값 전달이 어디서 꼬이는 듯

                    newArea = line_area.iloc[i]
                    oldArea = line_area.iloc[key_table[j]]
                    end_area = endarea

                    d1 = (
                        (newArea.loc[["xmax", "ymax"]] - end_area.loc[["xmin", "ymin"]])
                        - (
                            end_area.loc[["xmin", "ymin"]]
                            - newArea.loc[["xmin", "ymin"]]
                        )
                    ).sum()
                    d2 = (
                        (oldArea.loc[["xmax", "ymax"]] - end_area.loc[["xmin", "ymin"]])
                        - (
                            end_area.loc[["xmin", "ymin"]]
                            - oldArea.loc[["xmin", "ymin"]]
                        )
                    ).sum()

                    newAreaMin = newArea.loc["xmin"], newArea.loc["ymin"]
                    newAreaMax = newArea.loc["xmax"], newArea.loc["ymax"]
                    oldAreaMin = oldArea.loc["xmin"], oldArea.loc["ymin"]
                    oldAreaMax = oldArea.loc["xmax"], oldArea.loc["ymin"]
                    endAreaMin = end_area.loc["xmin"], end_area.loc["ymin"]
                    endAreaMax = end_area.loc["xmax"], end_area.loc["ymax"]

                    d1 = (
                        (endAreaMin[0] - newAreaMin[0])
                        + (endAreaMin[1] - newAreaMin[1])
                        + (newAreaMax[0] - endAreaMax[0])
                        + (newAreaMax[1] - endAreaMax[1])
                    )
                    d2 = (
                        (endAreaMin[0] - oldAreaMin[0])
                        + (endAreaMin[1] - oldAreaMin[1])
                        + (oldAreaMax[0] - endAreaMax[0])
                        + (oldAreaMax[1] - endAreaMax[1])
                    )
                    # d1과 d2이 음수가 나오는 의미가..?

                    if d1 > d2:
                        key = i

                    elif d1 < d2:
                        key = key_table[j]

                    if key_table.get(j) != None:
                        # tempKey = key_table[key]
                        key_table[j] = key

                    else:
                        key_table[j] = key

                else:
                    key_table[j] = i

    return key_table


class findNetworkException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


def findNetwork(components: pd.DataFrame):
    circuits = pd.DataFrame()
    layer_count = 0

    try:
        start_comp = components[components.start.str.contains("V")]

        if len(start_comp) == 0:
            raise findNetworkException("+ 전압에서 시작되는 전기소자가 존재하지 않음")

        if len(start_comp) >= 2:  # 시작이 만약 2개 이상일 때
            components.drop(index=start_comp.index, inplace=True)
            raise findNetworkException("임시에러, 시작점이 2개임")

        else:
            components.drop(index=start_comp.index, inplace=True)
            components = components.sort_values(by="end_coord")

            circuits = pd.concat([circuits, start_comp])
            circuits.loc[start_comp.name, "layer"] = layer_count

            next_point = start_comp
            layer_count += 1

            total_components = len(components)
            check_comp_count = 0

            while check_comp_count != total_components:
                next_point = next_point.end.to_string(index=False)
                row = next_point[0]
                col = next_point[1:]

                next_point = components[
                    components.loc[:, "start"].apply(lambda x: x[1:] == col)
                ]
                components.drop(index=next_point.index, inplace=True)

                next_point["layer"] = layer_count
                circuits = pd.concat([circuits, next_point], axis=0)

                if len(next_point) > 1:
                    next_point = next_point.iloc[[0]]
                    # -> 또 검색 해야함..
                    # 만약 병렬된 곳에서 타고타고 들어갈 수도..

                layer_count += 1
                check_comp_count += 1

            if len(components) != 0:
                err_comp_name = []
                t = circuits.loc[:, ["name", "start", "end"]]

                for comp_name, component in components.iterrows():
                    # print(
                    #     t[
                    #         (t["start"].str[1:] == component["start"][1:])
                    #         | (t["end"].str[1:] == component["end"][1:])
                    #     ]
                    # )

                    err_comp_name.append(comp_name)

                raise findNetworkException(f"잘못 연결된 전기소자, {','.join(err_comp_name)}")

            if circuits.iloc[[-1]]["end"][0][0] != "V":
                raise findNetworkException(f"회로가 - 전원에 연결되지 않음")

            return circuits

    except findNetworkException as findNetworkEx:
        return {"message": str(findNetworkEx)}
