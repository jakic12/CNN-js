const {expect} = require(`chai`);
const {
  CNN,
  Layer,
  ActivationFunction,
  NetworkArchitectures,
} = require(`../cnn`);

var mnist = require("mnist");

var set = mnist.set(10, 10);

var trainingSet = set.training;
var testSet = set.test;

describe("Convolutional neural network", () => {
  it(`LeNet5 doesn't throw error`, () => {
    expect(() => new CNN(NetworkArchitectures.LeNet5)).not.to.throw();
  });

  it(`confirmShape`, () => {
    expect(() => {
      new CNN([
        new Layer.INPUT(12, 12, 2),
        new Layer.CONV(5, 5, 3, 10, 3, 2, 3),
      ]);
    }).not.to.throw();

    expect(() => {
      new CNN([new Layer.INPUT(10, 10, 3), new Layer.POOL(3, 3, 3, 4, 3)]);
    }).not.to.throw();

    expect(() => {
      new CNN([new Layer.INPUT(10, 10, 3), new Layer.POOL(3, 3, 3, 3, 3)]);
    }).to.throw();

    expect(() => {
      new CNN([
        new Layer.INPUT(12, 12, 2),
        new Layer.CONV(5, 5, 3, 10, 3, 2, 1),
      ]);
    }).to.throw();

    expect(() => {
      new CNN([new Layer.INPUT(12, 12, 2), new Layer.FC(10)]);
    }).to.throw();

    expect(() => {
      new CNN([
        new Layer.INPUT(12, 12, 2),
        new Layer.FLATTEN(12, 12, 2),
        new Layer.FC(10),
      ]);
    }).not.to.throw();

    expect(
      new CNN([
        new Layer.INPUT(12, 12, 2),
        new Layer.CONV(6, 6, 4, 4, 4, 2, 1),
        new Layer.FLATTEN(5, 5, 4),
        new Layer.FC(10),
      ]).weights.length,
      `weights array should be as long as shape`,
    ).to.equal(4);

    expect(
      new CNN([
        new Layer.INPUT(12, 12, 2),
        new Layer.CONV(6, 6, 4, 4, 4, 2, 1),
        new Layer.FLATTEN(5, 5, 4),
        new Layer.FC(10),
      ]).weights[1].length,
    ).to.equal(4);

    expect(
      new CNN([
        new Layer.INPUT(12, 12, 2),
        new Layer.CONV(6, 6, 4, 4, 4, 2, 1),
        new Layer.FLATTEN(5, 5, 4),
        new Layer.FC(10),
      ]).weights[1][0].length,
      `filter should be as deep as the previous layer`,
    ).to.equal(2);

    expect(
      new CNN([
        new Layer.INPUT(12, 12, 2),
        new Layer.CONV(6, 6, 4, 4, 4, 2, 1),
        new Layer.FLATTEN(5, 5, 4),
        new Layer.FC(10),
      ]).weights[1][0][0].length,
      `filter width should be the same as filter size`,
    ).to.equal(4);

    expect(
      new CNN([
        new Layer.INPUT(12, 12, 2),
        new Layer.CONV(6, 6, 4, 4, 4, 2, 1),
        new Layer.FLATTEN(5, 5, 4),
        new Layer.FC(10),
      ]).weights[3].length,
      `fc weight height should be the same as previous layer length`,
    ).to.equal(5 * 5 * 4);

    expect(
      new CNN([
        new Layer.INPUT(12, 12, 2),
        new Layer.CONV(6, 6, 4, 4, 4, 2, 1),
        new Layer.FLATTEN(5, 5, 4),
        new Layer.FC(10),
      ]).weights[3][0].length,
      `fc weight width should be the same as next layer length`,
    ).to.equal(10);
  });

  it(`error reduces`, function () {
    this.timeout(0);
    let cnn = new CNN(NetworkArchitectures.LeNet5);

    let input = [
      new Array(32).fill(0).map((_, i) =>
        new Array(32).fill(0).map((_, j) => {
          if (i < 28 && j < 28) {
            return trainingSet[0].input[i * 28 + j];
          } else {
            return 0;
          }
        }),
      ),
    ];

    expect(() => {
      cnn.backpropagate([1]);
    }).to.throw();

    expect(() => {
      cnn.forwardPropagate([1]);
    });

    const diff = (a, b) =>
      a.map((v, i) => {
        if (b[i] == v) {
          console.log(v + ` ` + b[i]);
        } else if (b[i] > v) {
          console.log(v + ` \u001b[32m` + b[i] + `\u001b[0m`);
        } else {
          console.log(v + ` \u001b[31m` + b[i] + `\u001b[0m`);
        }
      });

    let errArr = [];
    cnn.learningRate = -0.001;
    for (let iter = 0; iter < 10; iter++) {
      const prevOut = cnn.layers[cnn.layers.length - 1];
      let out = cnn.forwardPropagate(input);
      /*if (iter != 0) {
        diff(prevOut, out);
      }*/
      let err = cnn.getError(trainingSet[0].output);
      errArr.push(err);
      cnn.backpropagate(trainingSet[0].output, true);
    }
    errArr.map((t, i) => {
      if (i > 0) {
        expect(errArr[i - 1]).to.be.above(t);
      }
    });
  });
});
