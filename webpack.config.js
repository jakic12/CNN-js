const path = require("path");

module.exports = {
  entry: "./training_web_worker.js",
  output: {
    filename: "training_web_worker.js",
    path: path.resolve(__dirname, "dist"),
    //library: "module",
    //libraryTarget: "commonjs",
  },
};
