const fs = require("fs");
const {expect} = require(`chai`);
const {CNN} = require("../../cnn");
const imageProcessor = require("../../imageProcessor");
const {deepNormalize, deepMap, max} = require("../../math");
const {doesNotMatch} = require("assert");

describe("Experimental image refinery", () => {
  it("refine test", async function (done) {
    this.timeout(0);
    //open network
    const cnn = new CNN(
      JSON.parse(fs.readFileSync("test/experimental/Cifar-10-LeNet5.cnn")),
    );

    const image = deepNormalize(
      await imageProcessor.imageToArray("test/experimental/test_img.png", {
        x: 32,
        y: 32,
        z: 3,
      }),
      255,
    );

    const refined = cnn.refineImageToBe(
      image,
      new Array(10).fill(0).map((_, i) => (i === 4 ? 1 : 0)),
      8000,
      -1,
    );

    imageProcessor
      .arrayToImage(
        deepNormalize(
          deepMap(refined, x => Math.max(Math.min(x * 255, 255), 0)),
          max(refined),
        ),
        "test/experimental/output.jpg",
      )
      .then(() => done())
      .catch(e => console.log(e));
  });
});
