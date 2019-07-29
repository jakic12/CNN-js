const { expect } = require(`chai`)
var {
  matrixMultiply,
  matrixDot,
  transpose,
  convolute,
  doubleInverse,
  correlate,
  getDimension,
  maxPool,
  flattenDeep,
  matrixAdd,
  deepMap
} = require("../math");

describe('Basic math functions', () => {
    it(`deep map`, () => {
        expect(deepMap([
            [
                new Float32Array([1,2,3]),
                new Float32Array([3,2,1])
            ],
            [
                [1,2,3],
                [3,2,1]
            ]
        ],(v) => v+1)).to.eql([
            [
                new Float32Array([2,3,4]),
                new Float32Array([4,3,2])
            ],
            [
                [2,3,4],
                [4,3,2]
            ]
        ])
    })
    it(`matrix dimension recognition`, () => {
        let m1 = [1,2,3,4],
        m2 = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],
        m3 = [
            [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],
            [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],
            [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
        ]

        expect(getDimension(m1)).to.eql(1)
        expect(getDimension(m2)).to.eql(2)
        expect(getDimension(m3)).to.eql(3)
        expect(getDimension([m3])).to.eql(4)
    })
    describe(`matrix dot product`, () => {
        it(`throws error on invalid dimensions`, () => {
            expect(() => {
                matrixDot([[1,2]], [[1,2,3,4]])
            }).to.throw()

        })
        it(`multiplies`, () => {
            expect(matrixDot(
                [
                    [3, 1, 1, 4],
                    [5, 3, 2, 1],
                    [6, 2, 9, 5]
                ],
                [
                    [4, 9],
                    [6, 8],
                    [9, 7],
                    [7, 6]
                ])
            ).to.eql([
                new Float32Array([55,66]),
                new Float32Array([63,89]),
                new Float32Array([152,163])
            ])
         })
    })

    it(`transpose`, () => {
        expect(transpose([[1], [2], [3]])).to.eql([
            new Float32Array([1,2,3])
        ])

        expect(transpose([
            [11,12,13,14],
            [21,22,23,24]
        ])).to.eql([
            new Float32Array([11,21]),
            new Float32Array([12,22]),
            new Float32Array([13,23]),
            new Float32Array([14,24])
        ])
    })

    it(`matrix multiplication`, () => {
        expect(matrixMultiply([
                [1,2,3,4],
                [3,2,1,4],
                [5,2,3,6],
                [2,9,9,4]
            ],
            [
                [0,0,0,1],
                [2,3,0,0],
                [0,0,0,0],
                [1,2,0,0]
            ]
        )).to.eql([
            new Float32Array([0,0,0,4]),
            new Float32Array([6,6,0,0]),
            new Float32Array([0,0,0,0]),
            new Float32Array([2,18,0,0])
        ])

        expect(matrixMultiply([
            [1,2,3,4,5,6,10],
            [3,4,2,3,5,2,3]
        ],[
            [0,0,0,2,0,1,0],
            [0,1,0,0,2,0,0]
        ])).to.eql([
            new Float32Array([0,0,0,8,0 ,6,0]),
            new Float32Array([0,4,0,0,10,0,0])
        ])

        expect(matrixMultiply([
            [
                [1,2,3,4],
                [4,3,2,1],
                [2,1,0,2],
            ],[
                [1,2,3,4],
                [0,3,1,2],
                [5,2,1,0],
            ],
        ],[
            [
                [0,0,1,0],
                [0,1,0,2],
                [1,1,1,1],
            ],[
                [0,0,1,0],
                [0,1,0,0],
                [0,0,0,1],
            ],
        ])).to.eql([
            [
                new Float32Array([0,0,3,0]),
                new Float32Array([0,3,0,2]),
                new Float32Array([2,1,0,2]),
            ],[
                new Float32Array([0,0,3,0]),
                new Float32Array([0,3,0,0]),
                new Float32Array([0,0,0,0]),
            ],
        ])

        expect(() => {
            matrixMultiply([
                [1,2,3],
                [1,2,3]
            ],[
                [1,2],
                [1,2]
            ])
        }).to.throw()
    })

    it(`matrix addition`, () => {
        expect(matrixAdd([1,2,3], [3,2,1])).to.eql(new Float32Array([4,4,4]))

        expect(matrixAdd([
            [1,2,3],
            [0,0,0],
            [3,2,1]
        ],[
            [4,3,2],
            [1,1,1],
            [0,0,1]
        ])).to.eql([
            new Float32Array([5,5,5]),
            new Float32Array([1,1,1]),
            new Float32Array([3,2,2])
        ])

        expect(matrixAdd([[
            [1,2,3],
            [0,0,0],
            [3,2,1]
        ]],[[
            [4,3,2],
            [1,1,1],
            [0,0,1]
        ]])).to.eql([[
            new Float32Array([5,5,5]),
            new Float32Array([1,1,1]),
            new Float32Array([3,2,2])
        ]])
    })

    it(`double inverse`, () => {
        expect(doubleInverse([[
            [1,3,2,4],
            [2,2,3,4],
            [5,2,3,4]
        ],[
            [1,3,2,4],
            [2,2,3,4],
            [5,2,3,4]
        ]])).to.eql([[
            new Float32Array([4,3,2,5]),
            new Float32Array([4,3,2,2]),
            new Float32Array([4,2,3,1])
        ],[
            new Float32Array([4,3,2,5]),
            new Float32Array([4,3,2,2]),
            new Float32Array([4,2,3,1])
        ]])

        expect(doubleInverse([
            [1,3,2,4],
            [2,2,3,4],
            [5,2,3,4]
        ])).to.eql([
            new Float32Array([4,3,2,5]),
            new Float32Array([4,3,2,2]),
            new Float32Array([4,2,3,1])
        ])

        expect(doubleInverse([
            [
                [1,3,2,4],
                [2,2,3,4],
                [5,2,3,4]
            ],[
                [4,3,2,5],
                [4,3,2,2],
                [4,2,3,1]
            ]
        ])).to.eql([
            [
                new Float32Array([4,3,2,5]),
                new Float32Array([4,3,2,2]),
                new Float32Array([4,2,3,1])
            ],[
                new Float32Array([1,3,2,4]),
                new Float32Array([2,2,3,4]),
                new Float32Array([5,2,3,4])
            ]
        ])

        expect(doubleInverse([1,2,3,4])).to.eql(new Float32Array([4,3,2,1]))
    })

    it(`convolute`, () => {
        expect(convolute([[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]], [[[
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]]], 1, 1)).to.eql([[
            new Float32Array([-13, -20, -17]),
            new Float32Array([-18, -24, -18]),
            new Float32Array([13, 20, 17])
        ]])
        
        expect(convolute([[
            [1, 2, 3, 1, 2],
            [3, 4, 1, 2, 3],
            [1, 1, 2, 1, 4]
        ]], [[[
            [0, 0, -1],
            [0, 0, 1],
            [0, 2, 1],
        ]]], 1, 1)).to.eql([[
            new Float32Array([0, -2, -2, 2, -1]),
            new Float32Array([2, 7, 11, 4, 6]),
            new Float32Array([6, 12, 7, 7, 9]),
        ]])

        expect(convolute([[
            [1, 0, 2, 1],
            [3, 1, 2, 1],
            [3, 1, 1, 0]
        ]], [[[
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 2]
        ]]], 2, 1)).to.eql([[
            new Float32Array([0, 1]),
            new Float32Array([4, 5])
        ]])
    })

    it(`correlate`, () => {
      expect(correlate([[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
      ]],[[[
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
      ]]],1,1)).to.eql([[
        new Float32Array([-13, -20, -17]),
        new Float32Array([-18, -24, -18]),
        new Float32Array([13, 20, 17])
      ]]);

      expect(correlate([[
        [1, 2, 3, 1, 2],
        [3, 4, 1, 2, 3],
        [1, 1, 2, 1, 4]
      ]], [[[
          [1, 2, 0],
          [1, 0, 0],
          [-1, 0, 0]
      ]]], 1, 1)).to.eql([[
        new Float32Array([0, -2, -2, 2, -1]),
        new Float32Array([2, 7, 11, 4, 6]),
        new Float32Array([6, 12, 7, 7, 9])
      ]]);

      expect(correlate([[
        [1, 0, 2, 1],
        [3, 1, 2, 1],
        [3, 1, 1, 0]
      ]],[[[
        [2, 1, 0],
        [1, 0, 1],
        [0, 0, 0]
      ]]],2,1)).to.eql([[new Float32Array([0, 1]), new Float32Array([4, 5])]]);

      expect(correlate([
            [
                [2,1,2,1,2],
                [1,2,0,2,2],
                [1,0,2,0,1],
                [0,0,2,2,0],
                [2,1,0,0,1]
            ],[
                [2,2,0,0,2],
                [2,2,1,1,1],
                [2,1,0,1,0],
                [1,0,0,1,0],
                [0,2,0,0,1]
            ],[
                [2,2,0,0,2],
                [1,1,0,0,1],
                [2,2,1,2,1],
                [1,0,2,0,1],
                [1,2,2,0,0]
            ]
        ],[
            [
                [
                    [-1,1,0],
                    [0,-1,0],
                    [0,0,-1]
                ],[
                    [-1,-1,0],
                    [0,0,1],
                    [0,-1,-1]
                ],[
                    [-1,0,1],
                    [1,1,-1],
                    [1,1,-1]
                ]
            ],[
                [
                    [-1,-1,1],
                    [1,1,1],
                    [0,1,0]
                ],[
                    [-1,-1,1],
                    [1,0,1],
                    [-1,-1,0]
                ],[
                    [-1,0,-1],
                    [-1,-1,0],
                    [0,0,1]
                ]
            ]
        ], 2, 1,[1, 0])).to.eql([
            [
                new Float32Array([-5,-2,1]),
                new Float32Array([1,-6,2]),
                new Float32Array([-1,7,-3])
            ], [
                new Float32Array([3,1,1]),
                new Float32Array([-1,0,-8]),
                new Float32Array([3,0,-2])
            ]
        ])
    });

    it(`maxPooling`, () => {
        expect(maxPool([[
            [1,1,2,4],
            [5,6,7,8],
            [3,2,1,0],
            [1,2,3,4]
        ]], 2, 2)).to.eql([[
            new Float32Array([6,8]),
            new Float32Array([3,4])
        ]])
    })



    it(`flattenDeep`, () =>{
        expect(flattenDeep([1,2,3,[1,2,3,4, [2,3,4]]])).to.eql([1, 2, 3, 1, 2, 3, 4, 2, 3, 4])
    })

})