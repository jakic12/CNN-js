const {
  CNN,
  Layer,
  ActivationFunction,
  NetworkArchitectures
} = require(`../cnn`);

var mnist = require("mnist");

describe(`training test`, () => {
  it(`should train the network`, () => {
    const set = mnist.set(1, 1);
    const trainingSet = set.training;
    const testSet = set.test;

    const trainingSetInputs = trainingSet.map(example => [
      new Array(32).fill(0).map((_, i) =>
        new Array(32).fill(0).map((_, j) => {
          if (i < 28 && j < 28) {
            return example.input[i * 28 + j];
          } else {
            return 0;
          }
        })
      )
    ]);

    //let cnn = new CNN(NetworkArchitectures.LeNet5);
    let cnn = new CNN([
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
      new Layer.FC(10, ActivationFunction.TANH, ActivationFunction.DTANH)
      //new Layer.FC(10, ActivationFunction.TANH, ActivationFunction.DTANH)
    ]);
    cnn.learningRate = -0.1;
    console.log(trainingSet.length, trainingSet[0]);
    trainingSetInputs.map((i, index) => {
      console.log(cnn.forwardPropagate(i));
      console.log(trainingSet[index].output);
    });
    let errArr = [];
    for (let epoch = 0; epoch < 20; epoch++) {
      let error = 0;
      for (let example = 0; example < trainingSet.length; example++) {
        for (let iter = 0; iter < 10; iter++) {
          const out = cnn.forwardPropagate(trainingSetInputs[example]);
          cnn.backpropagate(trainingSet[0].output);
          const err = cnn.getError(trainingSet[0].output);
          error += err;
          console.log(epoch, iter, err);
        }
      }
      errArr.push(error);

      console.log(`epoch:`, epoch, error / (10 * trainingSet.length));
    }
    trainingSetInputs.map((i, index) => {
      console.log(cnn.forwardPropagate(i));
      console.log(trainingSet[index].output);
    });
  });
});
