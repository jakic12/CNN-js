const {CNN} = require("./cnn");

self.addEventListener("message", message => {
  const neuralNet = new CNN(JSON.parse(message.data.network));
  const eventFnct = (eventName, d) => {
    self.postMessage({
      event: eventName,
      data: d,
      network: JSON.stringify(neuralNet),
    });
  };

  neuralNet.sgd(
    Object.assign(message.data.trainingProps, {
      onProgress: (...d) => eventFnct("batchProgress", d),
      onEnd: (...d) => eventFnct("end", d),
    }),
  );
});
