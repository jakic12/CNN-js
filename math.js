var GPU = require('gpu.js').GPU;
const gpu = new GPU(/*{ mode: 'dev' }*/);

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
    if (a.length != b.length && ( !a[0][0] || a[0].length != b[0].length ))
        throw new Error(`invalid dimensions, both arrays should have equal dimensions`)

    return gpu.createKernel(function (a1, b1) {
        return a1[this.thread.y][this.thread.x] * b1[this.thread.y][this.thread.x]
    }).setOutput([a.length, a[0].length])(a, b);
}

const transpose = a => {
    if(a[0] && a[0][0] && a[0][0][0])
        throw new Error(`transpose supports up to 2d arrays`)

    if(!a[0].length)
        a = [a]

    let d = [ a.length, a[0].length || 1 ] 
    
    return gpu.createKernel(function (a1) {
        return a1[this.thread.x][this.thread.y]
    }).setConstants({ is2D: a[0] != undefined })
    .setOutput(d)(a)
}

const doubleInverse = a => {
    let outX = a[0].length
    let outY = a.length

    return gpu.createKernel(function(a1) {
      return a1[
        this.constants.outY - this.thread.y - 1
      ][this.constants.outX - this.thread.x - 1];
    })
      .setConstants({ outX, outY })
      .setOutput([outX, outY])(a);
    
}

const convolute = (a, f, s = 1, p = 0) => {
    return correlate(a, doubleInverse(f), s, p)
}

const correlate = (a, f, s = 1, p = 0) => {
    const outY = parseInt(( a.length - f.length + 2 * p)/s + 1)
    const outX = parseInt(( a[0].length - f[0].length + 2 * p)/s + 1)
    
    return gpu.createKernel(function (a1, f1) {
        let sum = 0
        for (let y = 0; y < this.constants.fY; y++)
            for (let x = 0; x < this.constants.fX; x++){
                let y1 = y + this.thread.y * this.constants.s - this.constants.p
                let x1 = x + this.thread.x * this.constants.s - this.constants.p
    
                if (y1 >= 0 && y1 < this.constants.ySize && x1 >= 0 && x1 < this.constants.xSize)
                    sum += a1[y1][x1] * f1[y][x]
            }
        return sum
    }).setConstants({ xSize: a[0].length, ySize: a.length, s, p, fY: f.length, fX: f[0].length })
    .setOutput([outX, outY])(a, f)
}

class debugGpu{
    constructor(f){
        this.f = f;
    }

    mainLoop(){
        if(this.outY){
            return new Array(this.outY).fill(0).map((_, y) => 
                new Float32Array(this.outX).fill(0).map((__, x) => {
                    let strF = this.f.toString().replace(new RegExp("this.thread.x", 'g'), x).replace(new RegExp("this.thread.y", 'g'), y)
                    for (const key of Object.keys(this.constants)) {
                        strF = strF.replace(new RegExp(`this.constants.${key}`, 'g'),this.constants[key]);
                    }
                    let test = eval(`(${strF})(...${JSON.stringify([...arguments])})`);
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
        }else{
            this.outX = a[0]
            this.outY = a[1]
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
  correlate
};