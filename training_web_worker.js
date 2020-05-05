const {CNN} = require("./cnn");

self.addEventListener("message", message => {
  if (!message.data.cmd) {
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
  } else {
    if (message.data.cmd === `confusionMatrix`) {
      const neuralNet = new CNN(JSON.parse(message.data.network));
      const cm = neuralNet.confusionMatrix(message.data.dataset);
      self.postMessage({event: "confusionMatrix", data: cm});
    }
  }
});
