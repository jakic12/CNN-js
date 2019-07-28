var GPU = require('gpu.js').GPU;
const gpu = new GPU(/*{ mode: 'dev' }*/);

const getDimension = a => {
    const r = (a1, i) => {
        if(a1.length){
            return r(a1[0], i+1)
        }else{
            return i
        }
    }

    return r(a, 0)
}

/**
 * Dot product between 2 2D matrices
 * @param {Array<Array<Number>>} a 
 * @param {Array<Array<Number>>} b 
 */
const matrixDot = (a, b) => {
    if(a[0].length != b.length)
        throw new Error(`invalid dimensions a -> x (${a[0].length}) should equal b -> y (${b.length})`)
    return gpu.createKernel(function (a1, b1) {
        var sum = 0;
        for (var i = 0; i < this.constants.aWidth; i++) {
            sum += a1[this.thread.y][i] * b1[i][this.thread.x];
        }
        return sum;
    }).setConstants({ aWidth: a[0].length })
    .setOutput([b[0].length,a.length])(a,b);
}

const matrixMultiply = (a,b) => {
    if(getDimension(a) == 3 && getDimension(b) == 3){
        if (a.length != b.length || a[0].length != b[0].length || a[0][0].length != b[0][0].length )
            throw new Error(`invalid dimensions, both arrays should have equal shape`)

        return gpu.createKernel(function (a1, b1) {
            return a1[this.thread.z][this.thread.y][this.thread.x] * b1[this.thread.z][this.thread.y][this.thread.x]
        }).setOutput([a[0][0].length, a[0].length, a.length])(a, b);
    }else if(getDimension(a) == 2 && getDimension(b) == 2){
        if (a.length != b.length || a[0].length != b[0].length )
            throw new Error(`invalid dimensions, both arrays should have equal shape`)

        return gpu.createKernel(function (a1, b1) {
            return a1[this.thread.y][this.thread.x] * b1[this.thread.y][this.thread.x]
        }).setOutput([a[0].length, a.length])(a, b);

    }else if(getDimension(a) == 1 && getDimension(b) == 1){
        if (a.length != b.length)
            throw new Error(`invalid dimensions, both arrays should have equal shape`)

        return gpu.createKernel(function (a1, b1) {
            return a1[this.thread.x] * b1[this.thread.x]
        }).setOutput([a.length])(a, b);
    }else{
        throw new Error(`invalid array dimension`)
    }
}

const transpose = a => {
    if(getDimension(a) > 2)
        throw new Error(`transpose supports up to 2d arrays`)

    if(!a[0].length)
        a = [a]

    let d = [ a.length, a[0].length || 1 ] 
    
    return gpu.createKernel(function (a1) {
        return a1[this.thread.x][this.thread.y]
    }).setOutput(d)(a)
}

const doubleInverse = a => {
    if(getDimension(a) == 4){
        let outW = a.length
        let outZ = a[0].length
        let outY = a[0][0].length
        let outX = a[0][0][0].length 

        return new Array(outW).fill(0).map((_, w) =>
            gpu.createKernel(function(a1){
                return a1
                    [this.thread.z]
                    [this.constants.outY - this.thread.y - 1]
                    [this.constants.outX - this.thread.x - 1]
            }).setConstants({ outX, outY })
              .setOutput([ outX, outY, outZ ])(a[w])
        )
    }else if(getDimension(a) == 3){
        let outZ = a.length
        let outY = a[0].length
        let outX = a[0][0].length

        return gpu.createKernel(function(a1){
            return a1
                [this.thread.z]
                [this.constants.outY - this.thread.y - 1]
                [this.constants.outX - this.thread.x - 1]
        }).setConstants({ outX, outY })
          .setOutput([ outX, outY, outZ ])(a)

    }else if(getDimension(a) == 2){
        let outX = a[0].length
        let outY = a.length

        return gpu.createKernel(function(a1) {
          return a1[
            this.constants.outY - this.thread.y - 1
          ][this.constants.outX - this.thread.x - 1];
        })
          .setConstants({ outX, outY })
          .setOutput([outX, outY])(a);
    }else if(getDimension(a) == 1){
        let outX = a.length

        return gpu.createKernel(function(a1){
            return a1[this.constants.outX - this.thread.x - 1]
        }).setConstants({ outX })
          .setOutput([ outX ])(a)
    }else{
        throw new Error(`invalid array dimension`)
    }
    
}

const convolute = (a, f, s = 1, p = 0, b = null) => {
    return correlate(a, doubleInverse(f), s, p, b)
}

/**
 * correlates a 3D input with a 4D filter with a 3D output
 * @param {Array<Array<Array<Number>>>} a input array
 * @param {Array<Array<Array<Array<Number>>>>} f array of filters
 * @param {Number} s stride
 * @param {Number} p zero padding
 * @param {Array<Number>} b bias (array of biases for each output layer)
 */
const correlate = (a, f, s = 1, p = 0, b = null) => {
    if(getDimension(a) == 3 && getDimension(f) == 4){
        if(f[0].length != a.length){
            throw new Error(`filter depth doesnt match input depth`)
        }

        const outZ = f.length
        const outY = parseInt(( a[0].length - f[0][0].length + 2 * p)/s + 1)
        const outX = parseInt(( a[0][0].length - f[0][0][0].length + 2 * p)/s + 1)

        return new Array(outZ).fill(0).map((_, Tz) => gpu.createKernel(function (a1, f1, b1) {
            let sum = b1

            for (let z = 0; z < this.constants.fZ; z++)
                for (let y = 0; y < this.constants.fY; y++)
                    for (let x = 0; x < this.constants.fX; x++){
                        let y1 = y + this.thread.y * this.constants.s - this.constants.p
                        let x1 = x + this.thread.x * this.constants.s - this.constants.p
                    
                        if (y1 >= 0 && y1 < this.constants.ySize && x1 >= 0 && x1 < this.constants.xSize)
                            sum += a1[z][y1][x1] * f1[z][y][x]
                    }

            return sum
        })
          .setConstants({
            xSize: a[0][0].length, //input x size
            ySize: a[0].length, //input y size
            s, //stride
            p, //padding
            fY: f[0][0].length, //filter y size
            fX: f[0][0][0].length, //filter x size
            fZ: f[0].length, //filter z size

          })
          .setOutput([outX, outY])(a, f[Tz], b? b[Tz] : 0)
        )
    }else{
        throw new Error(`invalid array dimension (${getDimension(a)}, ${getDimension(f)})`)
    }
}

class debugGpu{
    constructor(f){
        this.f = f;
    }

    mainLoop(){
        if(this.outZ){
            return new Array(this.outZ).fill(0).map((_, z) => 
                new Array(this.outY).fill(0).map((__, y) =>
                    new Float32Array(this.outX).fill(0).map((__, x) => {
                        let strF = this.f.toString().replace(new RegExp("this.thread.x", 'g'), x).replace(new RegExp("this.thread.y", 'g'), y).replace(new RegExp("this.thread.z", 'g'), z)
                        for (const key of Object.keys(this.constants)) {
                            if(this.constants[key].length){
                                strF = strF.replace(new RegExp(`this.constants.${key}`, 'g'),`[${this.constants[key]}]`);
                            }else{
                                strF = strF.replace(new RegExp(`this.constants.${key}`, 'g'),this.constants[key]);
                            }
                        }
                        let strFunction = `(${strF})(...${JSON.stringify([...arguments])})`
                        console.log(strFunction)
                        let test = eval(strFunction)
                        return test
                    })
                )
            )
        }else if(this.outY){
            return new Array(this.outY).fill(0).map((_, y) => 
                new Float32Array(this.outX).fill(0).map((__, x) => {
                    let strF = this.f.toString().replace(new RegExp("this.thread.x", 'g'), x).replace(new RegExp("this.thread.y", 'g'), y)
                    for (const key of Object.keys(this.constants)) {
                        if(this.constants[key].length){
                            strF = strF.replace(new RegExp(`this.constants.${key}`, 'g'),`[${this.constants[key]}]`);
                        }else{
                            strF = strF.replace(new RegExp(`this.constants.${key}`, 'g'),this.constants[key]);
                        }
                    }
                    let strFunction = `(${strF})(...${JSON.stringify([...arguments])})`
                    console.log(strFunction)
                    let test = eval(strFunction)
                    return test
                })
            )
        }else{

        }
    }

    setConstants(arr){
        this.constants = {}

        for (const key of Object.keys(arr)) {
          this.constants[key] = arr[key];
        }

        return this
    }

    setOutput(a){
        if(a.length == 1){
            this.outX = a[0]
        }else if(a.length == 2){
            this.outX = a[0]
            this.outY = a[1]
        }else{
            this.outX = a[0]
            this.outY = a[1]
            this.outZ = a[2]
        }

        return this.mainLoop.bind(this)
    }
}

module.exports = {
  matrixMultiply,
  matrixDot,
  transpose,
  convolute,
  doubleInverse,
  correlate,
  getDimension
};