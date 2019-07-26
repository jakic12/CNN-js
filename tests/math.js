const { expect } = require(`chai`)
var {
  matrixMultiply,
  matrixDot,
  transpose,
  convolute,
  doubleInverse,
  correlate
} = require("../math");

describe('Basic math functions', () => {
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
    })

    it(`double inverse`, () => {
        expect(doubleInverse([
            [1,3,2,4],
            [2,2,3,4],
            [5,2,3,4]
        ])).to.eql([
            new Float32Array([4,3,2,5]),
            new Float32Array([4,3,2,2]),
            new Float32Array([4,2,3,1])
        ])
    })

    it(`convolute`, () => {
        expect(convolute([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], 1, 1)).to.eql([
            new Float32Array([-13, -20, -17]),
            new Float32Array([-18, -24, -18]),
            new Float32Array([13, 20, 17])
        ])
        
        expect(convolute([
            [1, 2, 3, 1, 2],
            [3, 4, 1, 2, 3],
            [1, 1, 2, 1, 4]
        ], [
            [0, 0, -1],
            [0, 0, 1],
            [0, 2, 1],
        ], 1, 1)).to.eql([
            new Float32Array([0, -2, -2, 2, -1]),
            new Float32Array([2, 7, 11, 4, 6]),
            new Float32Array([6, 12, 7, 7, 9]),
        ])

        expect(convolute([
            [1, 0, 2, 1],
            [3, 1, 2, 1],
            [3, 1, 1, 0]
        ], [
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 2]
        ], 2, 1)).to.eql([
            new Float32Array([0, 1]),
            new Float32Array([4, 5])
        ])
    })

    it(`correlate`, () => {
      expect(correlate([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
      ],[
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
      ],1,1)).to.eql([
        new Float32Array([-13, -20, -17]),
        new Float32Array([-18, -24, -18]),
        new Float32Array([13, 20, 17])
      ]);

      expect(correlate([
        [1, 2, 3, 1, 2],
        [3, 4, 1, 2, 3],
        [1, 1, 2, 1, 4]
      ], [
          [1, 2, 0],
          [1, 0, 0],
          [-1, 0, 0]
      ], 1, 1)).to.eql([
        new Float32Array([0, -2, -2, 2, -1]),
        new Float32Array([2, 7, 11, 4, 6]),
        new Float32Array([6, 12, 7, 7, 9])
      ]);

      expect(correlate([
        [1, 0, 2, 1],
        [3, 1, 2, 1],
        [3, 1, 1, 0]
      ],[
        [2, 1, 0],
        [1, 0, 1],
        [0, 0, 0]
      ],2,1)).to.eql([new Float32Array([0, 1]), new Float32Array([4, 5])]);
    });


})