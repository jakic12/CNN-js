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

module.exports = {
  matrixMultiply,
  matrixDot,
  transpose,
  convolute,
  doubleInverse,
  correlate,
  getDimension,
  // maxPool,
  // flattenDeep,
  matrixAdd,
  deepMap
  // backPropagateCorrelation,
  // update2Dmatrix
};
