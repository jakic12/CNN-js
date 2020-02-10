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
 * @param {Array<Number>} b bias (array of biases for each output layer)
 */
const correlate = (a, f, s = 1, p = 0, b = null) => {};

module.exports = {
  matrixMultiply,
  matrixDot,
  transpose,
  // convolute,
  doubleInverse,
  // correlate,
  getDimension,
  // maxPool,
  // flattenDeep,
  matrixAdd,
  deepMap
  // backPropagateCorrelation,
  // update2Dmatrix
};
