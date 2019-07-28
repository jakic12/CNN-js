const { expect } = require(`chai`)
const {
    CNN,
    Layer,
    ActivationFunction,
    NetworkArchitectures
} = require(`../cnn`)

describe('Convolutional neural network', () => {
    it(`LeNet5 doesn't throw error`, () => {
        expect(() => new CNN(NetworkArchitectures.LeNet5)).not.to.throw()
    })

    it(`confirmShape`, () => {
        expect(() => {
            new CNN([ new Layer.INPUT(12, 12, 2), new Layer.CONV(5, 5, 3, 10, 3, 2, 3) ])
        }).not.to.throw()

        expect(() => {
            new CNN([ new Layer.INPUT(10, 10, 3), new Layer.POOL(3, 3, 3, 4, 3) ])
        }).not.to.throw()

        expect(() => {
            new CNN([ new Layer.INPUT(10, 10, 3), new Layer.POOL(3, 3, 3, 3, 3) ])
        }).to.throw()

        expect(() => {
            new CNN([ new Layer.INPUT(12, 12, 2), new Layer.CONV(5, 5, 3, 10, 3, 2, 1) ])
        }).to.throw()
    })
})