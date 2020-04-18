const {
  CNN,
  Layer,
  ActivationFunction,
  NetworkArchitectures,
} = require(`../cnn`);

const { openDatasetFromBuffer } = require(`../datasetProcessor`);
const fs = require(`fs`);

const { expect } = require(`chai`);

const { softmax, maxIndex } = require(`../math`);

const dataset = openDatasetFromBuffer(fs.readFileSync(`test/data_batch_1.bin`));
var mnist = require("mnist");

describe(`training test`, () => {
  it(`should train the network`, function () {
    this.timeout(0);
    const set = mnist.set(10, 9);
    const trainingSet = /*set.training;*/ dataset.labels
      .map((e, i) =>
        i < 100
          ? { output: new Array(10).fill(0).map((_e, i) => (i == e ? 1 : 0)) }
          : undefined
      )
      .filter((e) => e !== undefined);

    const testSet = /*set.test;*/ dataset.labels
      .map((e, i) =>
        i < 200 && i > 190
          ? { output: new Array(10).fill(0).map((_e, i) => (i == e ? 1 : 0)) }
          : undefined
      )
      .filter((e) => e !== undefined);
    console.log(trainingSet, testSet);
    const trainingSetInputs = /*trainingSet.map((example) => [
      new Array(32).fill(0).map((_, i) =>
        new Array(32).fill(0).map((_, j) => {
          if (i < 28 && j < 28) {
            return example.input[i * 28 + j];
          } else {
            return 0;
          }
        })
      ),
    ]);*/ dataset.inputArrays.filter(
      (_a, i) => i < 100
    );

    const testSetInputs = /*testSet.map((example) => [
      new Array(32).fill(0).map((_, i) =>
        new Array(32).fill(0).map((_, j) => {
          if (i < 28 && j < 28) {
            return example.input[i * 28 + j];
          } else {
            return 0;
          }
        })
      ),
    ]);*/ dataset.inputArrays.filter(
      (_a, i) => i < 200 && i > 190
    );

    //let cnn = new CNN(NetworkArchitectures.LeNet5);
    let cnn = new CNN([
      new Layer.INPUT(32, 32, 3),
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
    ]);
    cnn.learningRate = -0.01;
    //console.log(trainingSet.length, trainingSet[0]);
    /*trainingSetInputs.map((i, index) => {
      console.log(cnn.forwardPropagate(i));
      console.log(trainingSet[index].output);
    });*/
    let errArr = [];
    for (let epoch = 0; epoch < 10; epoch++) {
      let error = 0;
      for (let example = 0; example < trainingSet.length; example++) {
        for (let iter = 0; iter < 1; iter++) {
          const out = cnn.forwardPropagate(trainingSetInputs[example]);
          cnn.backpropagate(trainingSet[example].output);
          cnn.updateWeights();
          const err = cnn.getError(trainingSet[example].output);
          error += err;
          //console.log(epoch, iter, err);
        }
      }
      errArr.push(error / (10 * trainingSet.length));

      console.log(`epoch:`, epoch, error / (10 * trainingSet.length));
    }
    errArr.map((_, i) => console.log(i, `,`, errArr[i]));
    trainingSetInputs.map((i, index) => {
      const netOut = cnn.forwardPropagate(i);
      console.log(`normal`, netOut);
      console.log(
        `softmax`,
        softmax(netOut).map((x) => Math.round(x * 100) / 100)
      );
      console.log(`expected`, trainingSet[index].output);
      expect([index, maxIndex(netOut)]).to.eql([
        index,
        maxIndex(trainingSet[index].output),
      ]);
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
