const { imageToArray, arrayToImage } = require(`../imageProcessor`);
const { deepNormalize, deepMap } = require(`../math`);

describe("Image processor tests", () => {
  it(`Image processor test converts to array`, async () => {
    console.log(
      deepNormalize(
        await imageToArray(`test/testImage.jpg`, { x: 100, y: 100 })
      )
    );
  });

  it(`Image processor correctly does the 2 way transformation`, async () => {
    await arrayToImage(
      deepMap(
        deepNormalize(
          await imageToArray(`test/testImage.jpg`, { x: 100, y: 100 })
        ),
        (e) => e * 255
      ),
      `test/output.jpg`
    );
  });
});
