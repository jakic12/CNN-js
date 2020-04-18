const fs = require(`fs`);
const { openDatasetFromBuffer } = require(`../datasetProcessor`);
const { arrayToImage } = require(`../imageProcessor`);

describe("Dataset processor tests", () => {
  it(`Converts to array without errors`, function () {
    this.timeout(0);
    const buffer = fs.readFileSync(`test/data_batch_1.bin`);
    console.log(buffer);
    console.log(openDatasetFromBuffer(buffer));
  });

  it(`Save an image from the dataset`, function () {
    this.timeout(0);
    arrayToImage(
      openDatasetFromBuffer(fs.readFileSync(`test/data_batch_1.bin`))
        .inputArrays[5],
      `test/datasetOutput.jpg`
    );
  });
});
