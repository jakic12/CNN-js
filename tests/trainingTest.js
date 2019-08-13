const {
    CNN,
    Layer,
    ActivationFunction,
    NetworkArchitectures
} = require(`../cnn`)

var mnist = require('mnist')

var set = mnist.set(10, 10);

var trainingSet = set.training;
var testSet = set.test;

let cnn = new CNN(NetworkArchitectures.LeNet5)
let errArr = []
for(let epoch = 0; epoch < 10; epoch++){
    let error = 0;
    for (let example = 0; example < trainingSet.length; example++) {
        const input = [new Array(32).fill(0).map((_, i) =>
            new Array(32).fill(0).map((_, j) => {
                if(i < 28 && j < 28){
                    return trainingSet[example].input[i*28 + j]
                }else{
                    return 0
                }
            })
        )]
        for(let iter = 0; iter < 100; iter++){
            console.log(epoch, example, iter)
            const out = cnn.forwardPropagate(input)
            cnn.backpropagate(trainingSet[0].output)
            const err = cnn.getError(trainingSet[0].output)
            error += err
        }
    }
    errArr.push(error)
    console.log(epoch, error)
}
console.log(asciichart.plot(errArr, { height: 5 }))