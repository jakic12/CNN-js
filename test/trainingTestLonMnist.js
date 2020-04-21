const {
  CNN,
  Layer,
  ActivationFunction,
  NetworkArchitectures,
} = require(`../cnn`);

const { expect } = require(`chai`);

const { softmax, maxIndex } = require(`../math`);

var mnist = require("mnist");

describe(`mnist training test`, () => {
  it(`should train the network to the mnist dataset`, function () {
    this.timeout(0);
    const set = mnist.set(100, 10);
    const trainingSet = set.training;
    const testSet = set.test;

    const trainingSetInputs = trainingSet.map((example) => [
      new Array(32).fill(0).map((_, i) =>
        new Array(32).fill(0).map((_, j) => {
          if (i < 28 && j < 28) {
            return example.input[i * 28 + j];
          } else {
            return 0;
          }
        })
      ),
    ]);

    trainingSet.forEach((t, i) => {
      t.input = trainingSetInputs[i];
    });

    const testSetInputs = testSet.map((example) => [
      new Array(32).fill(0).map((_, i) =>
        new Array(32).fill(0).map((_, j) => {
          if (i < 28 && j < 28) {
            return example.input[i * 28 + j];
          } else {
            return 0;
          }
        })
      ),
    ]);

    let cnn = new CNN(NetworkArchitectures.LeNet5);
    /*let cnn = new CNN([
      new Layer.INPUT(32, 32, 1),
      new Layer.CONV(
        28,
        28,
        6,
        5,
        6,
        1,
        0,
        ActivationFunction.TANH,
        ActivationFunction.DTANH
      ),
      new Layer.POOL(
        14,
        14,
        6,
        2,
        2,
        ActivationFunction.TANH,
        ActivationFunction.DTANH
      ),
      new Layer.CONV(
        10,
        10,
        16,
        5,
        16,
        1,
        0,
        ActivationFunction.TANH,
        ActivationFunction.DTANH
      ),
      new Layer.POOL(
        5,
        5,
        16,
        2,
        2,
        ActivationFunction.TANH,
        ActivationFunction.DTANH
      ),
      new Layer.CONV(
        1,
        1,
        10,
        5,
        10,
        1,
        0,
        ActivationFunction.TANH,
        ActivationFunction.DTANH
      ),
      new Layer.FLATTEN(1, 1, 10),
      new Layer.FC(10, ActivationFunction.TANH, ActivationFunction.DTANH),
      //new Layer.FC(10, ActivationFunction.TANH, ActivationFunction.DTANH)
    ]);*/

    let errArr = [];
    let successRates = [];
    cnn.sgd({
      learningRate: -0.05,
      epochs: 100,
      dataset: trainingSet,
      onProgress: (epoch, accuracy, error) => {
        errArr.push(error);
        successRates.push(accuracy);
        console.log(`epoch:`, epoch, error, `${accuracy * 100}%`);
      },
    });

    errArr.map((_, i) => console.log(i, `,`, errArr[i], `,`, successRates[i]));
    trainingSetInputs.map((i, index) => {
      const netOut = cnn.forwardPropagate(i);
      console.log(`normal`, netOut);
      console.log(
        `softmax`,
        softmax(netOut).map((x) => Math.round(x * 100) / 100)
      );
      console.log(`expected`, trainingSet[index].output);
      /*expect([index, maxIndex(netOut)]).to.eql([
        index,
        maxIndex(trainingSet[index].output),
      ]);*/
    });

    console.log(`Test set:`);
    const all = testSetInputs.length;
    let good = 0;
    testSetInputs.map((i, index) => {
      const netOut = cnn.forwardPropagate(i);
      console.log(`normal`, netOut);
      console.log(
        `softmax`,
        softmax(netOut).map((x) => Math.round(x * 100) / 100)
      );
      console.log(`expected`, testSet[index].output);
      console.log(
        `prediction`,
        index,
        maxIndex(netOut),
        maxIndex(testSet[index].output)
      );
      if (maxIndex(netOut) === maxIndex(testSet[index].output)) {
        good++;
      }
    });
    console.log(`test success rate: ${(good / all) * 100}%`);
  });
});
