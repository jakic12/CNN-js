const { imageToArray, arrayToImage } = require(`../imageProcessor`);

describe("Image processor tests", () => {
  it(`Image processor test converts to array`, async () => {
    console.log(await imageToArray(`test/testImage.jpg`, { x: 100, y: 100 }));
  });

  it(`Image processor correctly does the 2 way transformation`, async () => {
    await arrayToImage(
      await imageToArray(`test/testImage.jpg`, { x: 150, y: 100 }),
      `test/output.jpg`
    );
  });
});
