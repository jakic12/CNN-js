const GPU = require("gpu.js").GPU;
const gpuSettings = { mode: `cpu` };

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
  const gpu = new GPU(gpuSettings);
  if (a[0].length != b.length)
    throw new Error(
      `invalid dimensions a -> x (${a[0].length}) should equal b -> y (${b.length})`
    );
  return gpu
    .createKernel(function(a1, b1) {
      let sum = 0;
      for (let i = 0; i < this.constants.aWidth; i++) {
        sum += a1[this.thread.y][i] * b1[i][this.thread.x];
      }
      return sum;
    })
    .setConstants({ aWidth: a[0].length })
    .setOutput([b[0].length, a.length])(a, b);
};

const matrixMultiply = (a, b) => {
  const gpu = new GPU(gpuSettings);
  if (getDimension(a) == 3 && getDimension(b) == 3) {
    if (
      a.length != b.length ||
      a[0].length != b[0].length ||
      a[0][0].length != b[0][0].length
    )
      throw new Error(
        `invalid dimensions, both arrays should have equal shape`
      );

    return gpu
      .createKernel(function(a1, b1) {
        return (
          a1[this.thread.z][this.thread.y][this.thread.x] *
          b1[this.thread.z][this.thread.y][this.thread.x]
        );
      })
      .setOutput([a[0][0].length, a[0].length, a.length])(a, b);
  } else if (getDimension(a) == 2 && getDimension(b) == 2) {
    if (a.length != b.length || a[0].length != b[0].length)
      throw new Error(
        `invalid dimensions, both arrays should have equal shape`
      );

    return gpu
      .createKernel(function(a1, b1) {
        return (
          a1[this.thread.y][this.thread.x] * b1[this.thread.y][this.thread.x]
        );
      })
      .setOutput([a[0].length, a.length])(a, b);
  } else if (getDimension(a) == 1 && getDimension(b) == 1) {
    if (a.length != b.length)
      throw new Error(
        `invalid dimensions, both arrays should have equal shape`
      );

    return gpu
      .createKernel(function(a1, b1) {
        return a1[this.thread.x] * b1[this.thread.x];
      })
      .setOutput([a.length])(a, b);
  } else {
    throw new Error(`invalid array dimension`);
  }
};

const matrixAdd = (a, b) => {
  const gpu = new GPU(gpuSettings);
  if (getDimension(a) == 3 && getDimension(b) == 3) {
    if (
      a.length != b.length ||
      a[0].length != b[0].length ||
      a[0][0].length != b[0][0].length
    )
      throw new Error(
        `invalid dimensions, both arrays should have equal shape`
      );

    return gpu
      .createKernel(function(a1, b1) {
        return (
          a1[this.thread.z][this.thread.y][this.thread.x] +
          b1[this.thread.z][this.thread.y][this.thread.x]
        );
      })
      .setOutput([a[0][0].length, a[0].length, a.length])(a, b);
  } else if (getDimension(a) == 2 && getDimension(b) == 2) {
    if (a.length != b.length || a[0].length != b[0].length)
      throw new Error(
        `invalid dimensions, both arrays should have equal shape`
      );

    return gpu
      .createKernel(function(a1, b1) {
        return (
          a1[this.thread.y][this.thread.x] + b1[this.thread.y][this.thread.x]
        );
      })
      .setOutput([a[0].length, a.length])(a, b);
  } else if (getDimension(a) == 1 && getDimension(b) == 1) {
    if (a.length != b.length)
      throw new Error(
        `invalid dimensions, both arrays should have equal shape`
      );

    return gpu
      .createKernel(function(a1, b1) {
        return a1[this.thread.x] + b1[this.thread.x];
      })
      .setOutput([a.length])(a, b);
  } else {
    throw new Error(
      `invalid array dimension a:${getDimension(a)}, b:${getDimension(b)}`
    );
  }
};

const transpose = a => {
  const gpu = new GPU(gpuSettings);
  if (getDimension(a) > 2)
    throw new Error(`transpose supports up to 2d arrays`);

  if (!a[0].length) a = [a];

  let d = [a.length, a[0].length || 1];

  return gpu
    .createKernel(function(a1) {
      return a1[this.thread.x][this.thread.y];
    })
    .setOutput(d)(a);
};

const doubleInverse = a => {
  const gpu = new GPU(gpuSettings);
  if (getDimension(a) == 4) {
    let outW = a.length;
    let outZ = a[0].length;
    let outY = a[0][0].length;
    let outX = a[0][0][0].length;

    return new Array(outW).fill(0).map((_, w) =>
      gpu
        .createKernel(function(a1) {
          return a1[
            this.thread.z
          ][this.constants.outY - this.thread.y - 1][this.constants.outX - this.thread.x - 1];
        })
        .setConstants({ outX, outY })
        .setOutput([outX, outY, outZ])(a[w])
    );
  } else if (getDimension(a) == 3) {
    let outZ = a.length;
    let outY = a[0].length;
    let outX = a[0][0].length;

    return gpu
      .createKernel(function(a1) {
        return a1[
          this.thread.z
        ][this.constants.outY - this.thread.y - 1][this.constants.outX - this.thread.x - 1];
      })
      .setConstants({ outX, outY })
      .setOutput([outX, outY, outZ])(a);
  } else if (getDimension(a) == 2) {
    let outY = a.length;
    let outX = a[0].length;

    return gpu
      .createKernel(function(a1) {
        return a1[
          this.constants.outY - this.thread.y - 1
        ][this.constants.outX - this.thread.x - 1];
      })
      .setConstants({ outX, outY })
      .setOutput([outX, outY])(a);
  } else if (getDimension(a) == 1) {
    let outX = a.length;

    return gpu
      .createKernel(function(a1) {
        return a1[this.constants.outX - this.thread.x - 1];
      })
      .setConstants({ outX })
      .setOutput([outX])(a);
  } else {
    throw new Error(`invalid array dimension`);
  }
};

const convolute = (a, f, s = 1, p = 0, b = null) => {
  return correlate(a, doubleInverse(f), s, p, b);
};

/**
 * correlates a 3D input with a 4D filter with a 3D output
 * @param {Array<Array<Array<Number>>>} a input array
 * @param {Array<Array<Array<Array<Number>>>>} f array of filters
 * @param {Number} s stride
 * @param {Number} p zero padding
 * @param {Array<Number>} b bias (array of biases for each output layer)
 */
const correlate = (a, f, s = 1, p = 0, b = null) => {
  const gpu = new GPU(gpuSettings);
  if (getDimension(a) == 3 && getDimension(f) == 4) {
    if (f[0].length != a.length) {
      throw new Error(`filter depth doesnt match input depth`);
    }

    const outZ = f.length;
    const outY = parseInt((a[0].length - f[0][0].length + 2 * p) / s + 1);
    const outX = parseInt((a[0][0].length - f[0][0][0].length + 2 * p) / s + 1);

    return new Array(outZ).fill(0).map((_, Tz) =>
      gpu
        .createKernel(function(a1, f1, b1) {
          let sum = b1;

          for (let z = 0; z < this.constants.fZ; z++)
            for (let y = 0; y < this.constants.fY; y++)
              for (let x = 0; x < this.constants.fX; x++) {
                let y1 =
                  y + this.thread.y * this.constants.s - this.constants.p;
                let x1 =
                  x + this.thread.x * this.constants.s - this.constants.p;

                if (
                  y1 >= 0 &&
                  y1 < this.constants.ySize &&
                  x1 >= 0 &&
                  x1 < this.constants.xSize
                )
                  sum += a1[z][y1][x1] * f1[z][y][x];
              }

          return sum;
        })
        .setConstants({
          xSize: a[0][0].length, //input x size
          ySize: a[0].length, //input y size
          s, //stride
          p, //padding
          fY: f[0][0].length, //filter y size
          fX: f[0][0][0].length, //filter x size
          fZ: f[0].length //filter z size
        })
        .setOutput([outX, outY])(a, f[Tz], b ? b[Tz] : 0)
    );
  } else {
    throw new Error(
      `invalid array dimension (${getDimension(a)}, ${getDimension(f)})`
    );
  }
};

/**
 * Calculate the change in the filter array
 * @param {Array<Array<Array<Array<Number>>>>} f filter
 * @param {Array<Array<Array<Number>>>} dOut delta of the output
 * @param {Array<Array<Array<Number>>>} input the input
 * @param {Number} s stride
 * @param {Number} p padding
 * @returns {Object} an object containing dF - delta of the filters, dI - delta of the inputs
 */
//original:
/*
const backPropagateCorrelation = (f, dOut, input, s, p) => {
  const gpu = new GPU(gpuSettings);
  if (getDimension(input) == 3 && getDimension(f) == 4) {
    if (f[0].length != input.length) {
      throw new Error(`filter depth doesnt match input depth`);
    }

    return {
      dF: new Array(f.length).fill(0).map((_, m) =>
        gpu
          .createKernel(function(dOut, input) {
            let sum = 0;

            for (let h = 0; h < this.constants.dOutY; h++) {
              for (let k = 0; k < this.constants.dOutX; k++) {
                let y1 =
                  this.thread.y + h * this.constants.s - this.constants.p;
                let x1 =
                  this.thread.x + k * this.constants.s - this.constants.p;
                if (
                  y1 >= 0 &&
                  y1 < this.constants.inputY &&
                  x1 >= 0 &&
                  x1 < this.constants.inputX
                ) {
                  sum +=
                    dOut[this.constants.m][h][k] * input[this.thread.z][y1][x1];
                }
              }
            }

            return sum;
          })
          .setConstants({
            s,
            p,
            m,
            dOutY: dOut[0].length,
            dOutX: dOut[0][0].length,
            inputX: input[0].length,
            inputY: input.length
          })
          .setOutput([f[m][0][0].length, f[m][0].length, f[m].length])(
          dOut,
          input
        )
      ),

      dI: new Array(input.length).fill(0).map((_, z) => {
        // reduce the z dimension, because gpujs cant work with 4d arrays

        // take all filters that contributed to input[z]
        const croppedFilter = f.map((_, n) => f[n][z]);

        return gpu
          .createKernel(function(dOut, filter) {
            let sum = 0;

            for (let n = 0; n < this.constants.dOutZ; n++) {
              for (let h = 0; h < this.constants.dOutY; h++) {
                for (let k = 0; k < this.constants.dOutX; k++) {
                  const x1 =
                    this.thread.x + k * this.constants.s - this.constants.p;
                  const y1 =
                    this.thread.y + h * this.constants.s - this.constants.p;

                  if (
                    x1 >= 0 &&
                    x1 < this.constants.fX &&
                    y1 >= 0 &&
                    y1 < this.constants.fY
                  ) {
                    sum += dOut[n][h][k] * filter[n][x1][y1];
                  }
                }
              }
            }

            return sum;
          })
          .setConstants({
            s,
            p,
            fY: f[0][0].length,
            fX: f[0][0][0].length,
            dOutZ: dOut.length,
            dOutY: dOut[0].length,
            dOutX: dOut[0][0].length
          })
          .setOutput([input[0][0].length, input[0].length])(
          dOut,
          croppedFilter
        );
      })
    };
  } else {
    throw new Error(
      `invalid array dimension (${getDimension(input)}, ${getDimension(f)})`
    );
  }
};
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
        dF[m][f_d] = new Array(f[m][f_d].length)
          .fill(0)
          .map((_, i) => new Array(f[m][f_d][i].length).fill(0));
        //console.log(m, f_d, f[m][f_d].length);
        for (let in_i = 0; in_i < input[f_d].length; in_i++) {
          for (let in_j = 0; in_j < input[f_d][in_i].length; in_j++) {
            for (let dOut_i = 0; dOut_i < dOut[m].length; dOut_i++) {
              for (let dOut_j = 0; dOut_j < dOut[m][dOut_i].length; dOut_j++) {
                //dOut[m][dOut_i][dOut_j]
                //     ^ this is important

                const f_i1 = in_i - dOut_i * s + p;
                const f_j1 = in_j - dOut_j * s + p;

                if (
                  f_i1 >= 0 &&
                  f_i1 < f[m][f_d].length &&
                  f_j1 >= 0 &&
                  f_j1 < f[m][f_d][f_i1].length
                )
                  dF[m][f_d][f_i1][f_j1] +=
                    dOut[m][dOut_i][dOut_j] * input[f_d][in_i][in_j];
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

    return {
      dF,
      dI
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
  const gpu = new GPU(gpuSettings);
  if (getDimension(a) == 3) {
    const outZ = a.length;
    const outY = parseInt((a[0].length - f) / s + 1);
    const outX = parseInt((a[0][0].length - f) / s + 1);

    if (coordinateMode) {
      return new Array(outZ).fill(0).map((_, z2) =>
        new Array(outY).fill(0).map((_, y2) =>
          new Array(outX).fill(0).map((_, x2) => {
            let first = true;
            let max = 0;
            let outCoords = {};
            for (let y = 0; y < f; y++) {
              for (let x = 0; x < f; x++) {
                let y1 = y + y2 * s;
                let x1 = x + x2 * s;

                if (a[z2][y1][x1] > max || first) {
                  max = a[z2][y1][x1];
                  outCoords.x = x1;
                  outCoords.y = y1;
                }

                if (first) first = false;
              }
            }
            return outCoords;
          })
        )
      );
    } else {
      return gpu
        .createKernel(function(a1) {
          let first = true;
          let max = 0;
          for (let y = 0; y < this.constants.f; y++)
            for (let x = 0; x < this.constants.f; x++) {
              let y1 = y + this.thread.y * this.constants.s;
              let x1 = x + this.thread.x * this.constants.s;

              if (a1[this.thread.z][y1][x1] > max || first)
                max = a1[this.thread.z][y1][x1];

              if (first) first = false;
            }
          return max;
        })
        .setConstants({ f, s })
        .setOutput([outX, outY, outZ])(a);
    }
  } else {
    throw new Error(`invalid array dimension (${getDimension(a)})`);
  }
};

/**
 * update matrix a with da multiplied by a coefficient
 * @param {Array<Array<Number>>} a matrix to be updated
 * @param {Array<Array<Number>>} dA delta a
 * @param {Number} c the coefficient for Da to be multiplied with
 */
const update2Dmatrix = (a, dA, c) => {
  const gpu = new GPU(gpuSettings);
  if (getDimension(a) > 2) {
    return new Array(a.length)
      .fill(0)
      .map((_, i) => update2Dmatrix(a[i], dA[i], c));
  } else if (getDimension(a) == 2) {
    if (a.length != dA.length || a[0].length != dA[0].length)
      throw new Error(`Matrix sizes don't match`);

    return gpu
      .createKernel(function(a1, dA1) {
        return (
          a1[this.thread.y][this.thread.x] +
          dA1[this.thread.y][this.thread.x] * this.constants.coefficient
        );
      })
      .setConstants({
        coefficient: c
      })
      .setOutput([a[0].length, a.length])(a, dA);
  } else {
    throw new Error(`matrix dimension should be at least 2`);
  }
};

const flattenDeep = arr1 =>
  arr1.reduce(
    (acc, val) => (val.length ? acc.concat(flattenDeep(val)) : acc.concat(val)),
    []
  );

class debugGpu {
  constructor(f) {
    this.f = f;
  }

  mainLoop() {
    if (this.outZ) {
      return new Array(this.outZ).fill(0).map((_, z) =>
        new Array(this.outY).fill(0).map((__, y) =>
          new Float32Array(this.outX).fill(0).map((__, x) => {
            let strF = this.f
              .toString()
              .replace(new RegExp("this.thread.x", "g"), x)
              .replace(new RegExp("this.thread.y", "g"), y)
              .replace(new RegExp("this.thread.z", "g"), z);
            for (const key of Object.keys(this.constants)) {
              if (this.constants[key].length) {
                strF = strF.replace(
                  new RegExp(`this.constants.${key}`, "g"),
                  `[${this.constants[key]}]`
                );
              } else {
                strF = strF.replace(
                  new RegExp(`this.constants.${key}`, "g"),
                  this.constants[key]
                );
              }
            }
            let strFunction = `(${strF})(...${JSON.stringify([...arguments])})`;
            // console.log(strFunction)
            let test = eval(strFunction);
            return test;
          })
        )
      );
    } else if (this.outY) {
      return new Array(this.outY).fill(0).map((_, y) =>
        new Float32Array(this.outX).fill(0).map((__, x) => {
          let strF = this.f
            .toString()
            .replace(new RegExp("this.thread.x", "g"), x)
            .replace(new RegExp("this.thread.y", "g"), y);
          for (const key of Object.keys(this.constants)) {
            if (this.constants[key].length) {
              strF = strF.replace(
                new RegExp(`this.constants.${key}`, "g"),
                `[${this.constants[key]}]`
              );
            } else {
              strF = strF.replace(
                new RegExp(`this.constants.${key}`, "g"),
                this.constants[key]
              );
            }
          }
          let strFunction = `(${strF})(...${JSON.stringify([...arguments])})`;
          // console.log(strFunction)
          let test = eval(strFunction);
          return test;
        })
      );
    } else {
    }
  }

  setConstants(arr) {
    this.constants = {};

    for (const key of Object.keys(arr)) {
      this.constants[key] = arr[key];
    }

    return this;
  }

  setOutput(a) {
    if (a.length == 1) {
      this.outX = a[0];
    } else if (a.length == 2) {
      this.outX = a[0];
      this.outY = a[1];
    } else {
      this.outX = a[0];
      this.outY = a[1];
      this.outZ = a[2];
    }

    return this.mainLoop.bind(this);
  }
}

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
  update2Dmatrix
};
