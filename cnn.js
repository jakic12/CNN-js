"use strict";

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
  deepMap,
  backPropagateCorrelation,
  update2Dmatrix,
  maxIndex,
  sum,
} = require("./math");

const sigm = x => 1 / (1 + Math.exp(-x));
const ActivationFunctions = {
  RELU: {norm: x => (x > 0 ? x : 0), derivative: x => (x > 0 ? 1 : 0)},
  SIGMOID: {
    norm: sigm,
    derivative: x => sigm(x) * (1 - sigm(x)), //x => x * (1 - x) /*dSigmoidWithoutSigmoid*/,
  },
  TANH: {norm: Math.tanh, derivative: x => 1 - Math.pow(Math.tanh(x), 2)},
};

/**
 * Convolutional neural network, recieves a shape, give it data, train in and save it
 */
class CNN {
  /**
   * The constructor takes in the shape of the network and
   * initalizes weights and filters corresponding to the shape
   * @param {Array<Layer>} shape Array of Layer instances or a serialized network
   */
  constructor(shape) {
    if (shape.shape) {
      CNN.confirmShape(shape.shape);
      this.initialShape = shape.initialShape.map(s => Object.assign({}, s));
      this.shape = CNN.applyActivationFunctionToShape(shape.initialShape);

      this.errorF = (expected, actual) => Math.pow(actual - expected, 2) / 2;
      this.dErrorF = (expected, actual) => actual - expected;

      this.learningRate = shape.learningRate;

      this.layers = shape.layers;
      this.dlayers = shape.dlayers;
      this.weights = shape.weights;
      this.biases = shape.biases;
      this.serializeParams = shape.serializeParams;
      if (shape.serializeParams) {
        shape.serializeParams.forEach(p => {
          this[p] = shape[p];
        });
      }
    } else {
      CNN.confirmShape(shape);
      this.initialShape = shape.map(s => Object.assign({}, s));
      this.shape = CNN.applyActivationFunctionToShape(shape);

      const xavier = (fan_in, fan_out) =>
        Math.random() * Math.sqrt(6 / (fan_in + fan_out));
      const kaiming = (fan_in, fan_out) =>
        (Math.random() * 2 - 1) * Math.sqrt(2 / fan_in);
      const randomBiasF = () => 0;
      this.errorF = (expected, actual) => Math.pow(actual - expected, 2) / 2;
      this.dErrorF = (expected, actual) => actual - expected;

      this.learningRate = -0.01;

      this.layers = new Array(shape.length).fill(0).map((_, i) => {
        if (
          shape[i].type == LayerType.FC ||
          shape[i].type == LayerType.FLATTEN
        ) {
          return new Array(shape[i].l).fill(0);
        } else {
          return new Array(shape[i].d)
            .fill(0)
            .map(() =>
              new Array(shape[i].h)
                .fill(0)
                .map(() => new Array(shape[i].w).fill(0)),
            );
        }
      });

      this.dlayers = []; // dlayers are filled on the backpropagation step

      this.weights = new Array(shape.length).fill(0).map((_, i) => {
        if (i != 0) {
          if (shape[i].type == LayerType.FC) {
            // if layer is FC or FLATTEN, init a weight matrix
            return new Array(shape[i - 1].l)
              .fill(0)
              .map(() =>
                new Array(shape[i].l)
                  .fill(0)
                  .map(l => xavier(shape[i - 1].l, shape[i].l)),
              );
          } else if (shape[i].type == LayerType.CONV) {
            // else initialize a new filter
            return new Array(shape[i].k)
              .fill(0)
              .map(() =>
                new Array(shape[i - 1].d)
                  .fill(0)
                  .map(() =>
                    new Array(shape[i].f)
                      .fill(0)
                      .map(() =>
                        new Array(shape[i].f)
                          .fill(0)
                          .map((_, l) =>
                            kaiming(
                              shape[i - 1].w * shape[i - 1].d * shape[i - 1].h,
                            ),
                          ),
                      ),
                  ),
              );
          }
        }
      });

      // init biases as the same sizes of their layers
      this.biases = new Array(shape.length).fill(0).map((_, i) => {
        if (i != 0) {
          if (shape[i].type == LayerType.FC) {
            return new Array(this.shape[i].l).fill(0).map(randomBiasF);
          } else {
            return new Array(this.shape[i].d).fill(0).map(randomBiasF);
          }
        }
      });
    }
  }

  static applyActivationFunctionToShape(shape) {
    return shape.map(s =>
      s.afName
        ? Object.assign(s, {
            af: ActivationFunctions[s.afName].norm,
            daf: ActivationFunctions[s.afName].derivative,
          })
        : s,
    );
  }

  sgd({
    dataset,
    epochs,
    learningRate = this.learningRate,
    decay = 0,
    onProgress,
    onEnd,
  }) {
    this.learningRate = learningRate;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let error = 0;
      let correct = 0;
      for (let example = 0; example < dataset.length; example++) {
        let out = this.forwardPropagate(dataset[example].input);
        this.backpropagate(dataset[example].output);
        this.updateWeights();
        const err = this.getError(dataset[example].output);
        error += err;

        if (maxIndex(out) === maxIndex(dataset[example].output)) {
          correct++;
        }
      }

      if (onProgress)
        onProgress(
          epoch,
          correct / dataset.length,
          error / dataset.length,
          this.learningRate,
        );
      this.learningRate = learningRate / (1 + decay * epoch);
    }

    if (onEnd) onEnd();
  }

  confusionMatrix(dataset) {
    const classCount = this.layers[this.layers.length - 1].length;
    const out = new Array(classCount)
      .fill(0)
      .map(() => new Array(classCount).fill(0));
    for (let i = 0; i < dataset.length; i++) {
      const netOut = this.forwardPropagate(dataset[i].input);
      if (dataset[i].output.length)
        out[maxIndex(dataset[i].output)][maxIndex(netOut)]++;
      else out[dataset[i].output][maxIndex(netOut)]++;
    }
    return out;
  }

  static confusionMatrixStats(cm) {
    const classCount = cm.length;
    const stats = {actual: []};
    const avgSum = {precision: 0, recall: 0, f1Score: 0};
    const nanCount = {precision: 0, recall: 0, f1Score: 0};

    for (let i = 0; i < classCount; i++) {
      const precision = cm[i][i] / sum(cm.map(k => k[i]));
      const recall = cm[i][i] / sum(cm[i]);
      stats.actual[i] = {
        precision,
        recall,
        f1Score: (2 * (precision * recall)) / (precision + recall),
      };
      if (!isNaN(stats.actual[i].precision))
        avgSum.precision += stats.actual[i].precision;
      else nanCount.precision += 1;

      if (!isNaN(stats.actual[i].recall))
        avgSum.recall += stats.actual[i].recall;
      else nanCount.recall += 1;

      if (!isNaN(stats.actual[i].f1Score))
        avgSum.f1Score += stats.actual[i].f1Score;
      else nanCount.f1Score += 1;
    }

    stats.avg = {
      precision: avgSum.precision / (classCount - nanCount.precision),
      recall: avgSum.recall / (classCount - nanCount.recall),
      f1Score: avgSum.f1Score / (classCount - nanCount.f1Score),
      accuracy: sum(cm.map((k, i) => k[i])) / sum(cm),
    };

    return stats;
  }

  /**
   * Pass the data trough all layers and return the last one
   * @param {Array<Array<Array<Number>>>} data
   */
  forwardPropagate(data) {
    if (data.length != this.shape[0].d)
      throw new Error(
        `data depth (${data.length}) doesnt match required depth (${this.shape[0].d})`,
      );

    if (data[0].length != this.shape[0].h)
      throw new Error(
        `data height (${data[0].length}) doesnt match required height (${this.shape[0].h})`,
      );

    if (data[0][0].length != this.shape[0].w)
      throw new Error(
        `data width (${data[0][0].length}) doesnt match required width (${this.shape[0].w})`,
      );

    this.layers[0] = data;

    for (let i = 1; i < this.shape.length; i++) {
      switch (this.shape[i].type) {
        case LayerType.CONV:
          this.layers[i] = correlate(
            this.layers[i - 1],
            this.weights[i],
            this.shape[i].s,
            this.shape[i].p,
            this.biases[i],
          );
          break;
        case LayerType.POOL:
          this.layers[i] = maxPool(
            this.layers[i - 1],
            this.shape[i].f,
            this.shape[i].s,
          );
          break;
        case LayerType.FLATTEN:
          this.layers[i] = flattenDeep(this.layers[i - 1]);
          break;
        case LayerType.FC:
          this.layers[i] = matrixAdd(
            matrixDot([this.layers[i - 1]], this.weights[i])[0],
            this.biases[i],
          );
          break;
      }

      // Check for NaN before and after activation
      deepMap(this.layers[i], (x, i, v) => {
        if (isNaN(x)) throw new Error(`[${i}] output NaN before activation`);
        return x;
      });

      if (this.shape[i].af)
        this.layers[i] = deepMap(this.layers[i], x => this.shape[i].af(x));

      deepMap(this.layers[i], x => {
        if (isNaN(x)) throw new Error(`[${i}] output NaN after activation`);
        return x;
      });

      // console.log(i)
    }

    // return last layer
    return this.layers[this.layers.length - 1];
  }

  /**
   *
   * @param {Array} exp expected output
   * @param {boolean} returnArray does the function return the array?
   */
  getError(exp, returnArray = false) {
    if (exp.length != this.shape[this.shape.length - 1].l)
      throw new Error(
        `expected array length (${
          exp.length
        }) doesn't equal last layer length (${
          this.shape[this.shape.length - 1].l
        })`,
      );

    let dout = this.layers[this.shape.length - 1].map((v, j) =>
      this.errorF(exp[j], v),
    );
    this.error =
      dout.reduce((a, b) => a + b, 0) /
      this.layers[this.shape.length - 1].length;

    if (returnArray) return dout;
    else return this.error;
  }

  /**
   *
   * @param {Array} exp expected output
   */
  backpropagate(exp, updateWeights = false) {
    if (exp.length != this.shape[this.shape.length - 1].l)
      throw new Error(
        `expected array length (${
          exp.length
        }) doesn't equal last layer length (${
          this.shape[this.shape.length - 1].l
        })`,
      );

    for (let i = this.shape.length - 1; i > 0; i--) {
      if (this.shape[i].type == LayerType.FC) {
        if (i == this.shape.length - 1) {
          this.dlayers[i] = this.layers[i].map((v, j) =>
            this.dErrorF(exp[j], v),
          );
        } else {
          this.dlayers[i] = matrixDot(
            [this.dlayers[i + 1]],
            transpose(this.weights[i + 1]),
          )[0];
        }

        if (this.shape[i].daf)
          this.dlayers[i] = matrixMultiply(
            this.dlayers[i],
            deepMap(this.layers[i], v => this.shape[i].daf(v)),
          );

        if (updateWeights)
          for (let y = 0; y < this.weights[i].length; y++)
            for (let x = 0; x < this.weights[i][y].length; x++) {
              this.weights[i][y][x] +=
                this.layers[i - 1][y] * this.dlayers[i][x] * this.learningRate;
            }

        //TODO backpropagate for bias
      } else if (this.shape[i].type == LayerType.FLATTEN) {
        let darray;
        if (i == this.shape.length - 1) {
          darray = this.layers[i].map((v, j) => this.dErrorF(exp[j], v));
          //TODO test if FLATTEN layer works as a last layer
        } else {
          darray = matrixDot(
            [this.dlayers[i + 1]],
            transpose(this.weights[i + 1]),
          )[0];
        }

        if (this.shape[i + 1].daf)
          darray = matrixMultiply(
            darray,
            deepMap(this.layers[i], v => this.shape[i + 1].daf(v)),
          );

        this.dlayers[i] = darray;

        this.dlayers[i - 1] = new Array(this.shape[i].d)
          .fill(0)
          .map((_, i1) =>
            new Array(this.shape[i].h)
              .fill(0)
              .map((_, j) =>
                new Array(this.shape[i].w)
                  .fill(0)
                  .map(
                    (_, k) =>
                      darray[
                        i1 * this.shape[i].h * this.shape[i].w +
                          j * this.shape[i].h +
                          k
                      ],
                  ),
              ),
          );
      } else if (this.shape[i].type == LayerType.CONV) {
        const temp = backPropagateCorrelation(
          this.weights[i],
          this.dlayers[i],
          this.layers[i - 1],
          this.shape[i].s,
          this.shape[i].p,
        );
        const {dF, dI, dB} = temp;
        this.dlayers[i - 1] = dI;
        //console.log(dI.length, dI);

        // pass the derivatives trough the derivative of the activation function
        if (this.shape[i].daf)
          this.dlayers[i - 1] = matrixMultiply(
            this.dlayers[i - 1],
            deepMap(this.layers[i - 1], v => this.shape[i].daf(v)),
          );

        //update weights
        if (updateWeights)
          this.weights[i] = update2Dmatrix(
            this.weights[i],
            dF,
            this.learningRate,
          );

        //update biases
        if (updateWeights)
          this.biases[i] = this.biases[i].map(
            (b, i) => b + dB[i] * this.learningRate,
          );
      } else if (this.shape[i].type == LayerType.POOL) {
        let dIn = new Array(this.shape[i - 1].d)
          .fill(0)
          .map(() =>
            new Array(this.shape[i - 1].h)
              .fill(0)
              .map(() => new Array(this.shape[i - 1].w).fill(0)),
          );
        let maxCoords = maxPool(
          this.layers[i - 1],
          this.shape[i].f,
          this.shape[i].s,
          true,
        );

        for (let z = 0; z < this.shape[i].d; z++) {
          for (let y = 0; y < this.shape[i].h; y++) {
            for (let x = 0; x < this.shape[i].w; x++) {
              let coords = maxCoords[z][y][x];
              dIn[z][coords.y][coords.x] = this.dlayers[i][z][y][x];
            }
          }
        }

        this.dlayers[i - 1] = dIn;

        if (this.shape[i].daf)
          this.dlayers[i - 1] = matrixMultiply(
            this.dlayers[i - 1],
            deepMap(this.layers[i - 1], v => this.shape[i].daf(v)),
          );
      }

      deepMap(this.dlayers[i], x => {
        if (isNaN(x)) throw new Error(`[${i}] output ${x} after derivation`);
        return x;
      });
    }
  }

  updateWeights() {
    for (let i = this.shape.length - 1; i > 0; i--) {
      if (this.shape[i].type == LayerType.FC) {
        for (let y = 0; y < this.weights[i].length; y++)
          for (let x = 0; x < this.weights[i][y].length; x++) {
            this.weights[i][y][x] +=
              this.layers[i - 1][y] * this.dlayers[i][x] * this.learningRate;

            this.dlayers[i][x] = 0;
          }

        //TODO update bias
      } else if (this.shape[i].type == LayerType.CONV) {
        const temp = backPropagateCorrelation(
          this.weights[i],
          this.dlayers[i],
          this.layers[i - 1],
          this.shape[i].s,
          this.shape[i].p,
        );
        const {dF, dI, dB} = temp;
        this.weights[i] = update2Dmatrix(
          this.weights[i],
          dF,
          this.learningRate,
        );

        //update biases
        this.biases[i] = this.biases[i].map(
          (b, i) => b + dB[i] * this.learningRate,
        );
      }
    }
  }

  static confirmShape(shape) {
    if (shape[0].type != LayerType.INPUT)
      throw new Error(
        `the first layer isn't an input layer, instead is: ${shape[0].type}`,
      );
    for (let i = 1; i < shape.length; i++) {
      if (shape[i].type == LayerType.CONV) {
        if (
          shape[i].w !=
          (shape[i - 1].w - shape[i].f + 2 * shape[i].p) / shape[i].s + 1
        )
          throw new Error(
            `[${i}] CONV: outW doesn't equal to calculated outW expected: ${
              (shape[i - 1].w - shape[i].f + 2 * shape[i].p) / shape[i].s + 1
            }, actual: ${shape[i].w}`,
          );

        if (
          shape[i].h !=
          (shape[i - 1].h - shape[i].f + 2 * shape[i].p) / shape[i].s + 1
        )
          throw new Error(
            `[${i}] CONV: outH doesn't equal to calculated outH expected: ${
              (shape[i - 1].h - shape[i].f + 2 * shape[i].p) / shape[i].s + 1
            }, actual: ${shape[i].h}`,
          );

        if (shape[i].d != shape[i].k)
          throw new Error(
            `[${i}] CONV: number of kernels doesn't equal outD kernels: ${shape[i].k}, outD: ${shape[i].d}`,
          );
      } else if (shape[i].type == LayerType.POOL) {
        if (shape[i].w != (shape[i - 1].w - shape[i].f) / shape[i].s + 1)
          throw new Error(
            `[${i}] POOL: outW doesn't equal to calculated outW expected: ${
              (shape[i - 1].w - shape[i].f) / shape[i].s + 1
            }, actual: ${shape[i].w}`,
          );

        if (shape[i].h != (shape[i - 1].h - shape[i].f) / shape[i].s + 1)
          throw new Error(
            `[${i}] POOL: outH doesn't equal to calculated outH expected: ${
              (shape[i - 1].h - shape[i].f) / shape[i].s + 1
            }, actual: ${shape[i].h}`,
          );

        if (shape[i - 1].d != shape[i].d)
          throw new Error(
            `[${i}] POOL: outD doesn't equal inZ inZ: ${
              shape[i - 1].d
            }, outD: ${shape[i].d}`,
          );
      } else if (shape[i].type == LayerType.FC) {
        if (
          shape[i - 1].type != LayerType.FC &&
          shape[i - 1].type != LayerType.FLATTEN
        )
          throw new Error(
            `[${i}] FC: The previous layer should be type FC or FLATTEN`,
          );
      } else if (shape[i].type == LayerType.FLATTEN) {
        if (
          shape[i - 1].type == LayerType.FLATTEN ||
          shape[i - 1].type == LayerType.FC
        )
          throw new Error(`[${i}] FLATTEN: The previous layer can't be flat`);
      }
    }

    return true;
  }
}

const LayerType = {
  CONV: 0,
  POOL: 1,
  FC: 2,
  INPUT: 3,
  FLATTEN: 4,
};

const ActivationFunction = {
  RELU: `RELU`,
  SIGMOID: `SIGMOID`,
  TANH: `TANH`,
};

const Layer = {
  /**
   * The input layer of a network
   * @param {Number} w width of the input
   * @param {Number} h height of the input
   * @param {Number} d depth of the input
   */
  INPUT: function (w, h, d) {
    this.type = LayerType.INPUT;
    this.w = w;
    this.h = h;
    this.d = d;
  },
  /**
   * Convolution layer hyperparameters
   * @param {Number} w width of the output
   * @param {Number} h height of the output
   * @param {Number} d depth of the output
   * @param {Number} f filter size (x,y)
   * @param {Number} k number of filters
   * @param {Number} s stride
   * @param {Number} p zero padding
   * @param {function(Number):Number} afName activation function name
   */
  CONV: function (w, h, d, f, k, s, p, afName) {
    this.type = LayerType.CONV;
    this.w = w;
    this.h = h;
    this.d = d;
    this.f = f;
    this.k = k;
    this.s = s;
    this.p = p;
    this.afName = afName;
  },
  /**
   * Pooling layer hyperparameters
   * @param {Number} w width of the output
   * @param {Number} h height of the output
   * @param {Number} d depth of the output
   * @param {Number} f filter size
   * @param {Number} s stride
   * @param {function(Number):Number} afName activation function
   */
  POOL: function (w, h, d, f, s, afName) {
    this.type = LayerType.POOL;
    this.w = w;
    this.h = h;
    this.d = d;
    this.f = f;
    this.s = s;
    this.afName = afName;
  },
  /**
   * A fully connected layer
   * @param {Number} l length of the layer
   * @param {function(Number):Number} afName activation function name
   */
  FC: function (l, afName) {
    this.type = LayerType.FC;
    this.l = l;
    this.afName = afName;
  },
  /**
   * Convert a convolutional layerr to a fully connected layer
   * @param {Number} w width of the input
   * @param {Number} h height of the input
   * @param {Number} d depth of the input
   */
  FLATTEN: function (w, h, d) {
    this.type = LayerType.FLATTEN;
    this.w = w;
    this.h = h;
    this.d = d;
    this.l = w * h * d;
  },
};

const NetworkArchitectures = {
  LeNet5: [
    new Layer.INPUT(32, 32, 1),
    new Layer.CONV(28, 28, 6, 5, 6, 1, 0, ActivationFunction.TANH),
    new Layer.POOL(14, 14, 6, 2, 2, ActivationFunction.TANH),
    new Layer.CONV(10, 10, 16, 5, 16, 1, 0, ActivationFunction.TANH),
    new Layer.POOL(5, 5, 16, 2, 2, ActivationFunction.TANH),
    new Layer.CONV(1, 1, 120, 5, 120, 1, 0, ActivationFunction.TANH),
    new Layer.FLATTEN(1, 1, 120),
    //new Layer.FC(84, ActivationFunction.TANH),
    new Layer.FC(10, ActivationFunction.TANH),
  ],
  LeNet5Color: [
    new Layer.INPUT(32, 32, 3),
    new Layer.CONV(28, 28, 6, 5, 6, 1, 0, ActivationFunction.TANH),
    new Layer.POOL(14, 14, 6, 2, 2, ActivationFunction.TANH),
    new Layer.CONV(10, 10, 16, 5, 16, 1, 0, ActivationFunction.TANH),
    new Layer.POOL(5, 5, 16, 2, 2, ActivationFunction.TANH),
    new Layer.CONV(1, 1, 120, 5, 120, 1, 0, ActivationFunction.TANH),
    new Layer.FLATTEN(1, 1, 120),
    //new Layer.FC(84, ActivationFunction.TANH),
    new Layer.FC(10, ActivationFunction.TANH),
  ],
  // CustomReducedLeNet5Color: [
  //   new Layer.INPUT(32, 32, 3),
  //   new Layer.CONV(28, 28, 6, 5, 6, 1, 0, ActivationFunction.TANH),
  //   new Layer.POOL(14, 14, 6, 2, 2, ActivationFunction.TANH),
  //   new Layer.CONV(10, 10, 16, 5, 16, 1, 0, ActivationFunction.TANH),
  //   new Layer.POOL(5, 5, 16, 2, 2, ActivationFunction.TANH),
  //   new Layer.CONV(1, 1, 120, 5, 120, 1, 0, ActivationFunction.TANH),
  //   new Layer.FLATTEN(1, 1, 120),
  //   new Layer.FC(10, ActivationFunction.TANH),
  // ],
};

module.exports = {
  CNN,
  ActivationFunction,
  Layer,
  NetworkArchitectures,
  LayerType,
};
