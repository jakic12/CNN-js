const {vectorize, deepNormalize} = require(`./math`);

const openDatasetFromBuffer = (buffer, imageColorDepth = 3, imageSize = 32) => {
  const datasetArray = new Uint8Array(buffer);
  const imageChannelPixelCount = imageSize * imageSize;
  const imagePixelCount = imageChannelPixelCount * imageColorDepth;
  const rowCount = imagePixelCount + 1;
  const output = [];
  console.log(datasetArray.length / rowCount, imageColorDepth, imageSize);
  for (let n = 0; n < datasetArray.length / rowCount; n++) {
    output[n] = {};
    output[n].input = [];
    output[n].label = datasetArray[n * rowCount];
    for (let i = 0; i < imageColorDepth; i++) {
      output[n].input[i] = [];

      for (let j = 0; j < imageSize; j++) {
        output[n].input[i][j] = [];
        for (let k = 0; k < imageSize; k++) {
          output[n].input[i][j][k] =
            datasetArray[
              n * (imagePixelCount + 1) +
                i * imageChannelPixelCount +
                j * imageSize +
                k +
                1
            ];
        }
      }
    }
  }
  return output;
};

const vectorizeDatasetLabels = (dataset, outLength) => {
  dataset.forEach(c => {
    c.output = vectorize(c.label, outLength);
  });
  return dataset;
};

const uint8ArrayToString = buf => {
  const arr = new Uint8Array(buf);
  if (arr.length > 60000) {
    let outStr = "";
    for (let i = 0; i <= arr.length - 1; i += 60000) {
      outStr += String.fromCharCode.apply(null, arr.subarray(i, i + 60000));
    }
    return outStr;
  } else return String.fromCharCode.apply(null, arr);
};

const stringToUint8Array = str => {
  var buf = new ArrayBuffer(str.length);
  var bufView = new Uint8Array(buf);
  for (var i = 0, strLen = str.length; i < strLen; i++) {
    bufView[i] = str.charCodeAt(i);
  }
  return bufView;
};

const datasetToUint8Array = dataset => {
  const imageSize = dataset[0].input[0].length;
  const imageColorDepth = dataset[0].input.length;
  const imageChannelPixelCount = imageSize * imageSize;
  const imagePixelCount = imageChannelPixelCount * imageColorDepth;
  const rowCount = imagePixelCount + 1;

  const out = new Uint8Array(dataset.length * rowCount);
  for (let n = 0; n < dataset.length; n++) {
    out[n * rowCount] = dataset[n].label;
    for (let i = 0; i < imageColorDepth; i++) {
      for (let j = 0; j < imageSize; j++) {
        for (let k = 0; k < imageSize; k++) {
          out[
            n * (imagePixelCount + 1) +
              i * imageChannelPixelCount +
              j * imageSize +
              k +
              1
          ] = dataset[n].input[i][j][k];
        }
      }
    }
  }

  return out;
};

const normalizeInputData = (dataset, max) => {
  dataset.forEach((data, i) => {
    dataset[i].input = deepNormalize(dataset[i].input, max);
  });
  return dataset;
};

module.exports = {
  openDatasetFromBuffer,
  datasetToUint8Array,
  vectorizeDatasetLabels,
  stringToUint8Array,
  uint8ArrayToString,
  normalizeInputData,
};
