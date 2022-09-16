import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import schemdraw
import schemdraw.elements as e

def drawDiagram(V, circuit: list):
    R_name = [[element["name"] for element in level] for level in circuit]
    R_value = [[element["value"] for element in level] for level in circuit]

    d = schemdraw.Drawing()
    d.push()

    components = []
    for r in range(len(R_name)):
        if len(R_name[r]) > 1:
            if len(R_name[r]) == 2:  #저항 2개
                with schemdraw.Drawing(show=False) as parallelR :
                    parallelR += e.Line().right(parallelR.unit/4)
                    parallelR.push()

                    # 위쪽 저항
                    parallelR += e.Line().up(parallelR.unit/2)
                    parallelR += (R_1:= e.RES().right().label([str(R_name[r][0]), '\n' + str(R_value[r][0])+'$\Omega$']))
                    parallelR += e.Line().down(parallelR.unit/2)
                    parallelR.pop()

                    # 아래쪽
                    parallelR += e.Line().down(parallelR.unit/2)
                    parallelR += (R_2:= e.RES().right().label([str(R_name[r][1]), '\n' + str(R_value[r][1])+'$\Omega$']))
                    parallelR += e.Line().up(parallelR.unit/2)

                    # 직선
                    parallelR += e.Line().right(parallelR.unit/4)

                components.append(parallelR)

            if len(R_value[r]) == 3:   #저항 3개
                with schemdraw.Drawing(show=False) as parallelR :
                    
                    parallelR += e.Line().right(parallelR.unit/4)
                    parallelR.push()

                    # 위쪽
                    parallelR += e.Line().up(parallelR.unit/2)
                    parallelR += (R_1:= e.Resistor().right().label([str(R_name[r][0]), '\n' + str(R_value[r][0])+'$\Omega$']))
                    parallelR += e.Line().down(parallelR.unit/2)
                    parallelR.pop()

                    # 아래쪽
                    parallelR.push()
                    parallelR += e.Line().down(parallelR.unit/2)
                    parallelR += (R_2:= e.Resistor().right().label([str(R_name[r][2]), '\n' + str(R_value[r][2])+'$\Omega$']))
                    parallelR += e.Line().up(parallelR.unit/2)
                    parallelR.pop()

                    # 중간
                    parallelR += (R_3:= e.Resistor().label([str(R_name[r][1]), '\n' + str(R_value[r][1])+'$\Omega$']))

                    # 직선
                    parallelR += e.Line().right(parallelR.unit/4)

                    components.append(parallelR)
    #저항 1개
        elif len(R_value[r]) ==1:
            with schemdraw.Drawing(show=False) as p:        
                p += e.Resistor().label([str(R_name[r][0]), '\n' + str(R_value[r][0])+'$\Omega$'])
            components.append(p)
    #저항 없을 때        
    else:
        with schemdraw.Drawing(show=False) as l:
            l += e.Line().right()

    for c in components:
        d += e.ElementDrawing(c)

    d += (n1 := e.Dot())
    d += e.Line().down().at(n1.end)
    d += (n2 := e.Dot())
    d.pop()
    d += (n3 := e.Dot())
    d += e.SourceV().down().label(f"{V}V").at(n3.end).reverse()
    d += (n4 := e.Dot())
    d += e.Line().right().endpoints(n4.end,n2.end)

    r = d.get_imagedata('jpg')
    plt.clf()
    plt.close('all')
    return r

if __name__ == "__main__":
    V = 5
    circuit = [
        [	      
            {"name": "R10", "value": 3},
            {"name": "R11", "value": 3},
            {"name": "R11", "value": 3},
        ],
        [     
            {"name": "R21", "value": 2},
            {"name": "R21", "value": 2},
        ], 
        [
            {"name": "R30", "value": 6},
            {"name": "R31", "value": 6},
            {"name": "R31", "value": 6},
        ],
    ]

    drawDiagram(V, circuit)
