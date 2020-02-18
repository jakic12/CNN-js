const getDimension = a => {
  const r = (a1, i) => {
    if (a1.length) {
      return r(a1[0], i + 1);
    } else {
      return i;
    }
  };

  return r(a, 0);
};

/**
 * @callback deepMapCallback
 * @param {*} value
 * @param {Number} i
 * @param {Array} array
 */
/**
 *
 * @param {Array} a the multidimensional array
 * @param {deepMapCallback} f the function to be mapped
 */
const deepMap = (a, f) =>
  a.map((v, i, a1) => {
    if (v && v.length) {
      return deepMap(v, f);
    } else {
      return f(v, i, a1);
    }
  });

/**
 * Dot product between 2 2D matrices
 * @param {Array<Array<Number>>} a
 * @param {Array<Array<Number>>} b
 */
const matrixDot = (a, b) => {
  if (a[0].length != b.length)
    throw new Error(
      `invalid dimensions a -> x (${a[0].length}) should equal b -> y (${b.length})`
    );

  const out = [];
  for (let i = 0; i < a.length; i++) {
    // y
    out[i] = [];
    for (let j = 0; j < b[0].length; j++) {
      // x
      out[i][j] = 0;
      for (let j1 = 0; j1 < a[i].length; j1++) {
        out[i][j] += a[i][j1] * b[j1][j];
      }
    }
  }

  return out;
};

/**
 * Multiply two matrices of the same shape elementwise
 * @param {Array<Array<Number>>} a
 * @param {Array<Array<Number>>} b
 */
const matrixMultiply = (a, b) => {
  if (a.length != b.length) {
    throw new Error(`invalid dimensions, both arrays should have equal shape`);
  } else {
    if (a[0] instanceof Array && b[0] instanceof Array) {
      const out = [];
      for (let i = 0; i < a.length; i++) {
        out[i] = matrixMultiply(a[i], b[i]);
      }
      return out;
    } else if (a[0] instanceof Array != b[0] instanceof Array) {
      throw new Error(
        `invalid dimensions, both arrays should have equal shape`
      );
    } else {
      const out = [];
      for (let i = 0; i < a.length; i++) {
        out[i] = a[i] * b[i];
      }
      return out;
    }
  }
};

/**
 * Add two matrices of the same shape elementwise
 * @param {Array<Array<Number>>} a
 * @param {Array<Array<Number>>} b
 */
const matrixAdd = (a, b) => {
  if (a.length != b.length) {
    throw new Error(`invalid dimensions, both arrays should have equal shape`);
  } else {
    if (a[0] instanceof Array && b[0] instanceof Array) {
      const out = [];
      for (let i = 0; i < a.length; i++) {
        out[i] = matrixAdd(a[i], b[i]);
      }
      return out;
    } else if (a[0] instanceof Array != b[0] instanceof Array) {
      throw new Error(
        `invalid dimensions, both arrays should have equal shape`
      );
    } else {
      const out = [];
      for (let i = 0; i < a.length; i++) {
        out[i] = a[i] + b[i];
      }
      return out;
    }
  }
};

const transpose = a => {
  if (getDimension(a) > 2)
    throw new Error(`transpose supports up to 2d arrays`);

  if (!a[0].length) a = [a];

  const out = [];

  for (let i = 0; i < a[0].length; i++) {
    out[i] = [];
    for (let j = 0; j < a.length; j++) {
      out[i][j] = a[j][i];
    }
  }

  return out;
};

/**
 * Flips kernels (array of 3d kernels, flips only kernels)
 * @param {Array<Array<Array<Array<Number>>>>} a
 */
const doubleInverse = a => {
  if (getDimension(a) == 1) {
    return doubleInverse([[[a]]])[0][0][0];
  } else if (getDimension(a) == 2) {
    return doubleInverse([[a]])[0][0];
  } else if (getDimension(a) == 3) {
    return doubleInverse([a])[0];
  } else {
    const out = [];
    for (let f = 0; f < a.length; f++) {
      out[f] = [];

      for (let z = 0; z < a[f].length; z++) {
        out[f][z] = [];

        for (let y = 0; y < a[f][z].length; y++) {
          out[f][z][y] = [];

          for (let x = 0; x < a[f][z][y].length; x++) {
            out[f][z][y][x] =
              a[f][z][a[f][z].length - y - 1][a[f][z][y].length - x - 1];
          }
        }
      }
    }
    return out;
  }
};

/**
 * correlates a 3D input with a 4D filter with a 3D output
 * @param {Array<Array<Array<Number>>>} a input array
 * @param {Array<Array<Array<Array<Number>>>>} f array of filters
 * @param {Number} s stride
 * @param {Number} p zero padding
 * @param {Number} b array of biases (array of biases for each output layer)
 */
const correlate = (inputs, filters, stride = 1, padding = 0, b = null) => {
  if (filters[0].length != inputs.length) {
    throw new Error(
      `filter depth(${filters[0].length}) doesnt match input depth(${inputs.length})`
    );
  }

  if (filters[0][0].length != filters[0][0][0].length) {
    throw new Error(
      `filter should be a square matrix(${filters[0][0].length} != ${filters[0][0][0].length})`
    );
  }

  if (b && b.length != filters.length)
    throw new Error(
      `bias depth(${b.length}), should match output depth(${filters.length})`
    );

  const F = filters[0][0].length; // Filter height/width

  const D = inputs.length,
    H = parseInt((inputs[0].length - F + 2 * padding) / stride + 1), // output height
    W = parseInt((inputs[0][0].length - F + 2 * padding) / stride + 1); // output width

  return filters.map((filter, filterZ) => {
    const out = [];

    // for every output node
    for (let i = 0; i < H; i++) {
      out[i] = [];
      for (let j = 0; j < W; j++) {
        let sum = b ? b[filterZ] : 0;
        for (let z = 0; z < D; z++) {
          //for every node in filter
          for (let k = 0; k < F; k++) {
            for (let h = 0; h < F; h++) {
              // (h and k are filter coordinates)

              const i1 = i * stride + k - padding;
              const j1 = j * stride + h - padding;

              if (
                i1 >= 0 &&
                i1 < inputs[0].length &&
                j1 >= 0 &&
                j1 < inputs[0][0].length
              )
                sum += inputs[z][i1][j1] * filter[z][k][h];
            }
          }
        }
        out[i][j] = sum;
      }
    }

    return out;
  });
};

/**
 * This be the same as correlate, but with a flipped filter
 * @param {Array<Array<Array<Number>>>} a input array
 * @param {Array<Array<Array<Array<Number>>>>} f array of filters
 * @param {Number} s stride
 * @param {Number} p zero padding
 * @param {*} b
 */
const convolute = (a, f, s = 1, p = 0, b = null) => {
  return correlate(a, doubleInverse(f), s, p, b);
};

const max = a => {
  let max = -Infinity;
  deepMap(a, x => {
    if (x > max) {
      max = x;
    }
  });
  return max;
};

const sum = a => {
  let sum = 0;
  deepMap(a, x => {
    sum += x;
  });
  return sum;
};

const softmax = a => {
  const sum1 = sum(a);
  return deepMap(a, x => x / sum1);
};

const maxIndex = a => {
  if (getDimension(a) == 1) {
    let max = a[0];
    let index = 0;
    for (let i = 1; i < a.length; i++) {
      if (a[i] > max) {
        max = a[i];
        index = i;
      }
    }
    return index;
  } else {
    throw new Error(`maxIndex works only on 1d arrays`);
  }
};

/**
 * update matrix a with da multiplied by a coefficient
 * @param {Array<Array<Number>>} a matrix to be updated
 * @param {Array<Array<Number>>} dA delta a
 * @param {Number} c the coefficient for Da to be multiplied with
 */
const update2Dmatrix = (a, dA, c) =>
  a.map((v, i) => {
    if (v && v.length) {
      return update2Dmatrix(a[i], dA[i], c);
    } else {
      return a[i] + dA[i] * c;
    }
  });

/**
 *
 * @param {Array<Array<Array<Array<Number>>>>} f the filter used in the correlation
 * @param {Array<Array<Array<Number>>>} dOut derivative od the next layer
 * @param {Array<Array<Array<Number>>>} input the input used at the correlation
 * @param {Number} s stride
 * @param {Number} p padding
 */
const backPropagateCorrelation = (f, dOut, input, s, p) => {
  if (getDimension(input) == 3 && getDimension(f) == 4) {
    if (f[0].length != input.length) {
      throw new Error(`filter depth doesnt match input depth`);
    }

    //depths dont mix, so depth is the same for input and filter
    //create an array with the same dimensions as filter
    const dF = [];

    // m -> filter number
    for (let m = 0; m < f.length; m++) {
      dF[m] = [];
      for (let f_d = 0; f_d < f[m].length; f_d++) {
        dF[m][f_d] = [];
        //console.log(m, f_d, f[m][f_d].length);
        for (let f_i = 0; f_i < f[m][f_d].length; f_i++) {
          dF[m][f_d][f_i] = new Array(f[m][f_d][f_i].length).fill(0);
          for (let f_j = 0; f_j < f[m][f_d][f_i].length; f_j++) {
            for (let dOut_i = 0; dOut_i < dOut[m].length; dOut_i++) {
              for (let dOut_j = 0; dOut_j < dOut[m][dOut_i].length; dOut_j++) {
                //dOut[m][dOut_i][dOut_j]
                //     ^ this is important

                const in_i1 = dOut_i * s + p + f_i;
                const in_j1 = dOut_j * s + p + f_j;

                if (
                  in_i1 >= 0 &&
                  in_i1 < input[f_d].length &&
                  in_j1 >= 0 &&
                  in_j1 < input[f_d][in_i1].length
                )
                  dF[m][f_d][f_i][f_j] +=
                    dOut[m][dOut_i][dOut_j] * input[f_d][in_i1][in_j1];
              }
            }
          }
        }
      }
    }
    const dI = [];
    for (let m = 0; m < f.length; m++) {
      for (let in_d = 0; in_d < input.length; in_d++) {
        dI[in_d] = [];
        //note: f_d and in_d are the more or less the same thing
        for (let in_i = 0; in_i < input[in_d].length; in_i++) {
          dI[in_d][in_i] = new Array(input[in_d][in_i].length).fill(0);
          for (let in_j = 0; in_j < input[in_d][in_i].length; in_j++) {
            for (let dOut_i = 0; dOut_i < dOut[m].length; dOut_i++) {
              for (let dOut_j = 0; dOut_j < dOut[m][dOut_i].length; dOut_j++) {
                //coordinates of the filter value that affected input[in_i][in_j] => dOut[m][dOut_i][dOut_j]
                const f_i1 = in_i - dOut_i * s + p;
                const f_j1 = in_j - dOut_j * s + p;

                if (!dI[in_d][in_i]) dI[in_d][in_i] = [];

                if (
                  f_i1 >= 0 &&
                  f_i1 < f[m][in_d].length &&
                  f_j1 >= 0 &&
                  f_j1 < f[m][in_d][f_i1].length
                )
                  // prettier-ignore
                  dI[in_d][in_i][in_j] += dOut[m][dOut_i][dOut_j] * f[m][in_d][f_i1][f_j1];
              }
            }
          }
        }
      }
    }

    const dB = dOut.map(dOutM => sum(dOutM));

    return {
      dF,
      dI,
      dB
    };
  } else {
    throw new Error(
      `invalid array dimension (${getDimension(input)}, ${getDimension(f)})`
    );
  }
};

/**
 * Reduces x and y dimensions of an array by max pooling
 * @param {Array<Array<Array<Number>>>} a the input array
 * @param {Number} f filter size
 * @param {Number} s stride
 * @param {boolean} coordinateMode return only coordinates of the max number
 */
const maxPool = (a, f, s, coordinateMode = false) => {
  if (getDimension(a) == 3) {
    return a.map(layer2d => {
      const outY = (layer2d.length - f) / s + 1;
      const outX = (layer2d[0].length - f) / s + 1;

      let outLayer = [];
      for (let y = 0; y < outY; y++) {
        outLayer[y] = [];
        for (let x = 0; x < outX; x++) {
          let max = layer2d[y * s][x * s];
          let maxCoords = { x: x * s, y: y * s };

          for (let i = 0; i < f; i++) {
            for (let j = 0; j < f; j++) {
              let y1 = y * s + i;
              let x1 = x * s + j;

              if (layer2d[y1][x1] > max) {
                max = layer2d[y1][x1];
                maxCoords = { x: x1, y: y1 };
              }
            }
          }
          if (coordinateMode) outLayer[y][x] = maxCoords;
          else outLayer[y][x] = max;
        }
      }
      return outLayer;
    });
  } else {
    throw new Error(
      `invalid array dimension (${getDimension(a)}), should be 3`
    );
  }
};

/**
 * Converts an n-dimensional to a 1-dimensional array
 * @param {Array} arr1 n-dimensional array
 */
const flattenDeep = arr1 =>
  arr1.reduce(
    (acc, val) => (val.length ? acc.concat(flattenDeep(val)) : acc.concat(val)),
    []
  );

module.exports = {
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
  max,
  sum,
  softmax,
  maxIndex
};
