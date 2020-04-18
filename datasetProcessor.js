const openDatasetFromBuffer = (buffer, imageColorDepth = 3, imageSize = 32) => {
  const datasetArray = new Uint8Array(buffer);
  const imageChannelPixelCount = imageSize * imageSize;
  const imagePixelCount = imageChannelPixelCount * imageColorDepth;
  const rowCount = imagePixelCount + 1;
  const inputArrays = [];
  const labels = [];
  console.log(datasetArray.length / rowCount, imageColorDepth, imageSize);
  for (let n = 0; n < datasetArray.length / rowCount; n++) {
    inputArrays[n] = [];
    labels[n] = datasetArray[n * rowCount];
    for (let i = 0; i < imageColorDepth; i++) {
      inputArrays[n][i] = [];

      for (let j = 0; j < imageSize; j++) {
        inputArrays[n][i][j] = [];
        for (let k = 0; k < imageSize; k++) {
          inputArrays[n][i][j][k] =
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
  return { inputArrays, labels };
};

module.exports = {
  openDatasetFromBuffer,
};
