const fs = require(`fs`);
const {
  openDatasetFromBuffer,
  datasetToUint8Array,
  vectorizeDatasetLabels,
  stringToUint8Array,
  uint8ArrayToString,
} = require(`../datasetProcessor`);
const {arrayToImage} = require(`../imageProcessor`);
const {expect} = require(`chai`);
const {sum} = require(`../math`);

describe("Dataset processor tests", () => {
  it(`Converts array to string and back`, function () {
    const arr = new Uint8Array(new Array(100).fill(255));
    expect(stringToUint8Array(uint8ArrayToString(arr))).to.deep.equal(arr);
  });
  it(`Converts to array without errors`, function () {
    this.timeout(0);
    const buffer = fs.readFileSync(`test/data_batch_1.bin`);
    console.log(buffer);
    console.log(openDatasetFromBuffer(buffer));

    const uintArr = datasetToUint8Array(
      openDatasetFromBuffer(buffer).filter((_e, i) => i < 10),
    );

    /*console.log(uintArr.length);
    const outTest = uint8ArrayToString(uintArr);
    console.log(outTest.length);
    const back = stringToUint8Array(outTest);
    console.log(back.length);

    console.log(openDatasetFromBuffer(back));
    fs.writeFileSync(`test/datasetAsJson.bin`, JSON.stringify(outTest));*/
  });

  it(`Save an image from the dataset`, function () {
    this.timeout(0);
    arrayToImage(
      openDatasetFromBuffer(fs.readFileSync(`test/data_batch_1.bin`))[5].input,
      `test/datasetOutput.jpg`,
    );
  });

  it(`Converts two ways back to the original`, function () {
    this.timeout(0);
    const buffer = fs.readFileSync(`test/data_batch_1.bin`);
    const original = datasetToUint8Array(openDatasetFromBuffer(buffer));

    expect(original).to.deep.equal(new Uint8Array(buffer));
  });

  it(`vectorizes the labels`, function () {
    this.timeout(0);
    const result = vectorizeDatasetLabels(
      openDatasetFromBuffer(fs.readFileSync(`test/data_batch_1.bin`)),
      10,
    );

    result.forEach(c => {
      expect(sum(c.output)).to.equal(1);
      expect(c.output.length).to.equal(10);
      expect(c.output[c.label]).to.equal(1);
    });
  });
});
