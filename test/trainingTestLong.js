const {
  CNN,
  Layer,
  ActivationFunction,
  NetworkArchitectures,
} = require(`../cnn`);

const {
  openDatasetFromBuffer,
  vectorizeDatasetLabels,
} = require(`../datasetProcessor`);
const fs = require(`fs`);

const {expect} = require(`chai`);

const {softmax, maxIndex} = require(`../math`);

const dataset = vectorizeDatasetLabels(
  openDatasetFromBuffer(fs.readFileSync(`test/data_batch_1.bin`)),
  10,
);

describe(`training test`, () => {
  it(`should train the network`, function () {
    this.timeout(0);

    let trainingSet = dataset.filter((_e, i) => i < 100);
    let testSet = dataset.filter((_e, i) => i < 200 && i > 190);

    let cnn = new CNN([
      new Layer.INPUT(32, 32, 3),
      new Layer.CONV(28, 28, 6, 5, 6, 1, 0, ActivationFunction.TANH),
      new Layer.POOL(14, 14, 6, 2, 2, ActivationFunction.TANH),
      new Layer.CONV(10, 10, 16, 5, 16, 1, 0, ActivationFunction.TANH),
      new Layer.POOL(5, 5, 16, 2, 2, ActivationFunction.TANH),
      new Layer.CONV(1, 1, 120, 5, 120, 1, 0, ActivationFunction.TANH),
      new Layer.FLATTEN(1, 1, 120),
      //new Layer.FC(84, ActivationFunction.TANH),
      new Layer.FC(10, ActivationFunction.TANH),
    ]);
    let errArr = [];
    let successRates = [];

    cnn.sgd({
      learningRate: -0.01,
      epochs: 100,
      decay: 0.005,
      dataset: trainingSet,
      onProgress: (epoch, accuracy, err, learningRate) => {
        errArr.push(err);
        successRates.push(accuracy);
        console.log(
          `epoch:`,
          epoch,
          err,
          `lr:`,
          learningRate,
          `${accuracy * 100}%`,
        );
      },
    });

    errArr.map((_, i) => console.log(i, `,`, errArr[i], `,`, successRates[i]));

    trainingSet.map((i, index) => {
      const netOut = cnn.forwardPropagate(i.input);
      console.log(`normal`, netOut);
      console.log(
        `softmax`,
        softmax(netOut).map(x => Math.round(x * 100) / 100),
      );
      console.log(`expected`, trainingSet[index].output);
      expect([index, maxIndex(netOut)]).to.eql([
        index,
        maxIndex(trainingSet[index].output),
      ]);
    });

    console.log(`Test set:`);
    const all = trainingSet.length;
    let good = 0;
    testSet.map((i, index) => {
      const netOut = cnn.forwardPropagate(i.input);
      console.log(`normal`, netOut);
      console.log(
        `softmax`,
        softmax(netOut).map(x => Math.round(x * 100) / 100),
      );
      console.log(`expected`, testSet[index].output);
      console.log(
        `prediction`,
        index,
        maxIndex(netOut),
        maxIndex(testSet[index].output),
      );
      if (maxIndex(netOut) === maxIndex(testSet[index].output)) {
        good++;
      }
    });
    console.log(`test success rate: ${(good / all) * 100}%`);
  });
});
