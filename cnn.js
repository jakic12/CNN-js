class CNN{
    constructor(shape){
        CNN.confirmShape(shape)
        this.shape = shape

        const randomWeightF = () => Math.random()*2

        this.layers = new Array(shape.length).fill(0).map((_, i) => {
            if(shape[i].type == LayerType.FC || shape[i].type == LayerType.FLATTEN){
                return new Array(shape[i].l).fill(0)
            }else{
                return new Array(shape[i].d).fill(0).map(() => 
                    new Array(shape[i].h).fill(0).map(() =>
                        new Array(shape[i].w).fill(0)
                    )
                )
            }
        })

        this.weights = new Array(shape.length).fill(0).map((_, i) => {
            if(i != 0){
                if(shape[i].type == LayerType.FC){
                    if(shape[i-1].type == LayerType.FC || shape[i-1].type == LayerType.FLATTEN){
                        return new Array(shape[i-1].l).fill(0).map(() =>
                            new Array(shape[i].l).fill(0).map(randomWeightF)
                        )
                    }else{
                        return new Array(shape[i-1].d).fill(0).map(() =>
                            new Array(shape[i].l).fill(0).map(randomWeightF)
                        )
                    }
                }else if(shape[i].type == LayerType.CONV){
                    return new Array(shape[i].k).fill(0).map(() => 
                        new Array(shape[i-1].d).fill(0).map(() =>
                            new Array(shape[i].f).fill(0).map(() =>
                                new Array(shape[i].f).fill(0).map(randomWeightF)
                            )
                        )
                    )
                }
            }
        })

        console.log(this.weights)
    }

    static confirmShape(shape){
        if(shape[0].type != LayerType.INPUT)
            throw new Error(`the first layer isn't an input layer, instead is: ${shape[0].type}`)
        for(let i = 1; i < shape.length; i++){
            if(shape[i].type == LayerType.CONV){
                if(shape[i].w != (shape[i-1].w - shape[i].f + 2 * shape[i].p) / shape[i].s + 1)
                    throw new Error(`[${i}] CONV: outW doesn't equal to calculated outW expected: ${(shape[i-1].w - shape[i].f + 2 * shape[i].p) / shape[i].s + 1}, actual: ${shape[i].w}`)

                if(shape[i].h != (shape[i-1].h - shape[i].f + 2 * shape[i].p) / shape[i].s + 1)
                    throw new Error(`[${i}] CONV: outH doesn't equal to calculated outH expected: ${(shape[i-1].h - shape[i].f + 2 * shape[i].p) / shape[i].s + 1}, actual: ${shape[i].h}`)

                if(shape[i].d != shape[i].k)
                    throw new Error(`[${i}] CONV: number of kernels doesn't equal outD kernels: ${shape[i].k}, outD: ${shape[i].d}`)

            }else if(shape[i].type == LayerType.POOL){
                if(shape[i].w != (shape[i-1].w - shape[i].f) / shape[i].s + 1)
                    throw new Error(`[${i}] POOL: outW doesn't equal to calculated outW expected: ${(shape[i-1].w - shape[i].f) / shape[i].s + 1}, actual: ${shape[i].w}`)
                
                if(shape[i].h != (shape[i-1].h - shape[i].f) / shape[i].s + 1)
                    throw new Error(`[${i}] POOL: outH doesn't equal to calculated outH expected: ${(shape[i-1].h - shape[i].f) / shape[i].s + 1}, actual: ${shape[i].h}`) 
                
                if(shape[i-1].d != shape[i].d)
                throw new Error(`[${i}] POOL: outD doesn't equal inZ inZ: ${shape[i-1].d}, outD: ${shape[i].d}`)
            }else if(shape[i].type == LayerType.FC){

            }
        }

        return true
    }
}

const LayerType = {
    CONV: 0,
    POOL: 1,
    FC:2,
    INPUT:3,
    FLATTEN:4
}

const sigm = x => 1/(1+Math.exp(-x))
const ActivationFunction = {
    RELU: x => x>0? x : 0,
    DRELU: x => x>0? 1 : 0,
    SIGMOID: sigm,
    DSIGMOID: x => sigm(x)*(1 - sigm(x)),
    TANH: Math.tanh,
    DTANH: x => (1 - Math.pow(Math.tanh(x), 2))
}

const Layer = {
    /**
     * The input layer of a network
     * @param {Number} w width of the input
     * @param {Number} h height of the input
     * @param {Number} d depth of the input
     */
    INPUT: function(w, h, d){
        this.type = LayerType.INPUT
        this.w = w
        this.h = h
        this.d = d
    },
    /**
     * Convolution layer hyperparameters
     * @param {Number} w width of the output
     * @param {Number} h height of the output
     * @param {Number} d depth of the output
     * @param {Number} f filter size (x,y)
     * @param {Number} k number of filters
     * @param {Number} s stride
     * @param {Number} p zero padding
     * @param {Number} af activation function
     */
    CONV: function(w, h, d, f, k, s, p, af){
        this.type = LayerType.CONV
        this.w = w
        this.h = h
        this.d = d
        this.f = f
        this.k = k
        this.s = s
        this.p = p
        this.af = af
    },
    /**
     * Pooling layer hyperparameters
     * @param {Number} w width of the output
     * @param {Number} h height of the output
     * @param {Number} d depth of the output
     * @param {Number} f filter size
     * @param {Number} s stride
     * @param {Number} af activation function
     */
    POOL: function(w, h, d, f, s, af){
        this.type = LayerType.POOL
        this.w = w
        this.h = h
        this.d = d
        this.f = f
        this.s = s
        this.af = af
    },
    /**
     * A fully connected layer
     * @param {Number} l length of the layer
     * @param {Number} af activation function
     */
    FC: function(l, af){
        this.type = LayerType.FC
        this.l = l
        this.af = af
    },
    /**
     * Convert a convolutional layerr to a fully connected layer
     * @param {Number} w width of the input
     * @param {Number} h height of the input
     * @param {Number} d depth of the input
     */
    FLATTEN: function(w, h, d){
        this.type = LayerType.FLATTEN
        this.w = w
        this.h = h
        this.d = d
        this.l = w * h * d
    }
}

const NetworkArchitectures = {
    LeNet5: [
        new Layer.INPUT(32, 32, 1),
        new Layer.CONV(28, 28, 6, 5, 6, 1, 0, ActivationFunction.TANH),
        new Layer.POOL(14, 14, 6, 2, 2, ActivationFunction.TANH),
        new Layer.CONV(10, 10, 16, 5, 16, 1, 0, ActivationFunction.TANH),
        new Layer.POOL(5, 5, 16, 2, 2, ActivationFunction.TANH),
        new Layer.CONV(1, 1, 120, 5, 120, 1, 0, ActivationFunction.TANH),
        new Layer.FLATTEN(1, 1, 120),
        new Layer.FC(84, ActivationFunction.TANH),
        new Layer.FC(10, ActivationFunction.TANH)
    ]
}

module.exports = {
    CNN,
    ActivationFunction,
    Layer,
    NetworkArchitectures
}