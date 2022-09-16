import numpy as np


def calcCurrentAndVoltage(V, circuit):
    R_value = []
    for level in circuit:  # 합성저항 구하기
        R_value1 = []
        for element in level:
            R_value1.append(1 / int(element["value"]))
        R_value.append(R_value1)

    r = sum([1 / sum(r) for r in R_value])

    I = V / r

    i = []
    for level in circuit:  # 각 저항에 걸리는 전압
        R_voltage1 = []
        i0 = []
        for element in level:
            R_voltage1.append(int(element["value"]))
            level_sum_res = 1 / (sum([1 / R_vol for R_vol in R_voltage1]))
            i0 = [I * level_sum_res for R_vol in R_voltage1]
        i.append(i0)

    return r, I, i


if __name__ == "__main__":
    V = 10
    # circuit = [
    #     [
    #         {"name": "R10", "value": 3},
    #     ],
    #     [
    #         {"name": "R21", "value": 2},
    #     ],
    #     [
    #         {"name": "R30", "value": 6},
    #     ],
    # ]
    circuit = [
        [{"name": "R1", "value": 100}],
        [{"name": "R2", "value": 100}, {"name": "R3", "value": 100}],
        [{"name": "R4", "value": 100}],
    ]
    R_TH, I, NODE_VOL = calcCurrentAndVoltage(V, circuit)

    print(R_TH, I, NODE_VOL)
