class CNN{
    constructor(){

    }
}

let LayerType = {
    CONV: 0,
    ACTIV: 1,
    POOL: 2
}

let sigm = x => 1/(1+Math.exp(-x))
let ActivationFunction = {
    RELU: x => x>0? x : 0,
    DRELU: x => x>0? 1 : 0,
    SIGMOID: sigm,
    DSIGMOID: x => sigm(x)*(1 - sigm(x))
}

let Layer = {
    /**
     * Convolution layer hyperparameters
     * @param {Number} w width of the input
     * @param {Number} h height of the input
     * @param {Number} d depth of the input
     * @param {Number} f filter size (x,y)
     * @param {Number} k number of filters
     * @param {Number} s stride
     * @param {Number} p zero padding
     */
    CONV: function(w, h, d, f, k, s, p){
        this.type = LayerType.CONV
        this.w = w
        this.h = h
        this.d = d
        this.f = f
        this.k = k
        this.s = s
        this.p = p
    },
    /**
     * Activation layer
     * @param {Function} f the activation function
     */
    ACTIV: function(f){
        this.type = LayerType.ACTIV
        this.f = f
    },
    /**
     * Pooling layer hyperparameters
     * @param {Number} w width of the input
     * @param {Number} h height of the input
     * @param {Number} d depth of the input
     * @param {Number} f filter size
     * @param {Number} s stride
     */
    POOL: function(w, h, d, f, s){

    }
}

module.exports = {
    CNN
}