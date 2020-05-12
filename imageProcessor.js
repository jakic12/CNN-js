const Jimp = require("jimp");

/**
 *
 * @param {*} image image URL, buffer, path...
 * @param {Object} sizeDim output Array size {x:xSize, y:ySize, z:zSize}
 *      x and y are mandatory, z is optional and can be 3 or 1, default is 3
 *
 */
const imageToArray = (imageData, sizeDim) =>
  new Promise((resolve, reject) => {
    Jimp.read(imageData)
      .then(image => {
        let resizedImage = resizeImage(image, sizeDim);

        let out;
        if (!sizeDim.z || sizeDim.z === 3) {
          out = [[], [], []];
          for (let j = 0; j < sizeDim.y; j++) {
            out[0][j] = [];
            out[1][j] = [];
            out[2][j] = [];
            for (let k = 0; k < sizeDim.x; k++) {
              const pixel = Jimp.intToRGBA(resizedImage.getPixelColor(k, j));

              out[0][j][k] = pixel.r;
              out[1][j][k] = pixel.g;
              out[2][j][k] = pixel.b;
            }
          }
        } else {
          out = [[]];
          for (let j = 0; j < sizeDim.y; j++) {
            out[0][j] = [];
            for (let k = 0; k < sizeDim.x; k++) {
              const pixel = Jimp.intToRGBA(resizedImage.getPixelColor(k, j));
              out[0][j][k] = pixel.r;
            }
          }
        }

        resolve(out);
      })
      .catch(e => reject(e));
  });

/**
 *
 * @param {*} array 3d array
 * @param {*} writeTo optional, path to write to
 * @returns image buffer
 */
const arrayToImage = (array, writeTo) => {
  return new Promise(
    (resolve, reject) =>
      new Jimp(array[0][0].length, array[0].length, (err, image) => {
        for (let y = 0; y < array[0].length; y++) {
          for (let x = 0; x < array[0][y].length; x++) {
            if (array.length === 1) {
              image.setPixelColor(
                Jimp.rgbaToInt(
                  array[0][y][x],
                  array[0][y][x],
                  array[0][y][x],
                  1,
                ),
                x,
                y,
              );
            } else {
              image.setPixelColor(
                Jimp.rgbaToInt(
                  array[0][y][x],
                  array[1][y][x],
                  array[2][y][x],
                  1,
                ),
                x,
                y,
              );
            }
          }
        }

        if (writeTo) {
          image.write(writeTo, () =>
            image.getBuffer(Jimp.AUTO, b => resolve(b)),
          );
        } else {
          image.getBuffer(Jimp.AUTO, b => resolve(b));
        }
      }),
  );
};

const resizeImage = (image, sizeDim) => {
  if (sizeDim.z) {
    if (sizeDim.z === 1) {
      return resizeImage(
        image.greyscale(),
        Object.assign({}, sizeDim, {z: undefined}),
      );
    } else {
      return resizeImage(image, Object.assign({}, sizeDim, {z: undefined}));
    }
  } else {
    return image.resize(sizeDim.x, sizeDim.y);
  }
};

module.exports = {
  imageToArray,
  resizeImage,
  arrayToImage,
};
