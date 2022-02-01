/**
 *  ______ _______ _____  ___ ___ _______ _______ _______ _______ _______ _______ ______          _____ _______
 * |   __ \       |     ||   |   |     __|       |    |  |     __|       |   |   |   __ \______ _|     |     __|
 * |    __/   -   |       \     /|    |  |   -   |       |__     |   -   |   |   |    __/______|       |__     |
 * |___|  |_______|_______||___| |_______|_______|__|____|_______|_______|_______|___|         |_______|_______|
 *
 * © Daniel Berio (@colormotor) 2021 - ...
 *
 * Speed based brush dabbing method. Extends the method described in
 * @inproceedings {BerioEG2018
 * booktitle = {EG 2018 - Short Papers},
 * editor = {Diamanti, Olga and Vaxman, Amir},
 * title = {{Expressive Curve Editing with the Sigma Lognormal Model}},
 * author = {Berio, Daniel and Leymarie, Frederic Fol and Plamondon, RÃ©jean},
 * year = {2018},
 * publisher = {The Eurographics Association}
 * }
 *
 **/


"use strict"

const mth = require('./mth.js');

const brush = function() {}

// const hat = (x, sharpness) => {
//     return (mth.erf((1.-Math.abs(x*2))*sharpness)*0.5+0.5);
// }

const scurve2 = (t) => {
    return 1. / (1. + Math.exp(-Math.pow(t,3)))*2.0 - 1.0;
}

const hat = (x, sharpness) => {
    return (mth.erf((1.-Math.abs(x*2))*sharpness)*0.5+0.5);
}

brush.brush_image = (w, h, theta, power, smoothness, hole=2, hole_amt=0) => {
    const radius = Math.max(w, h);
    const size = parseInt(radius)*3;
    const center = [0, 0];
    let x = mth.linspace((-size/2)/radius, (size/2)/radius, size+1);
    let y = mth.linspace((-size/2)/radius, (size/2)/radius, size+1);
    //console.log(theta);
    const [ct, st] = [Math.cos(theta), Math.sin(theta)];
    //let xr = mth.sub(mth.mul(x, ct), mth.mul(y, st)); //  x * ct - y * st
    //let yr = mth.add(mth.mul(x, st), mth.mul(y, ct)); //  x * st + y * ct
    const ratio = 0.5;
    // scale
    //x = mth.div(x, w/radius);
    //y = mth.div(y, h/radius);
    const xdiv = w/radius;
    const ydiv = h/radius;
    // superellipse distance
    var v, xi, yi;
    let img = y.map(row=>mth.zeros(x.length));
    for (let i = 0; i < y.length; i++){
        for (let j = 0; j < x.length; j++){
            xi = (x[j]*ct - y[i]*st)/xdiv;
            yi = (x[j]*st + y[i]*ct)/ydiv;
            v = Math.sqrt(Math.abs(xi**power) +
                          Math.abs(yi**power));

            img[i][j] = hat(v, smoothness);
            if (hole > 0)
                img[i][j] *= (1 - hat(v*hole, smoothness)*hole_amt);
            img[i][j] = mth.clamp(img[i][j], 0, 1);
        }
    }

    return img;
}

const srgb = (color, mul) => {
    //return color;//
    const gamma = 1/2.2; //1.0/2.2; //1.0/2.2;
    return color.map(v=>Math.pow((v*mul)/255, gamma)*255);
}

brush.brush_image_p5 = (p5, color, alpha,
                        w, h, theta, power, smoothness, hole=2, hole_amt=0) => {
    const data = brush.brush_image(w, h, theta, power, smoothness, hole, hole_amt);

    let img = p5.createImage(data[0].length, data.length);
    img.loadPixels();
    let v, p;
    for (let i = 0; i < img.width; i++)
        for (let j = 0; j < img.height; j++){
            v = data[j][i]*alpha; //Math.pow(data[j][i], 2.2)*255;//*alpha;
            //v = (v > 128) ? 255: 0;
            //v = Math.pow(data[j][i], 1.0/2.2)*255;//*alpha;
            p = data[j][i];
            img.set(i, j, p5.color(...srgb(color, p), v)); //color[0]*p, color[1]*p, color[2]*p, v));
        }
    img.updatePixels();
    return img;
}

brush.width = (s, normalizer, minw, maxw) => {
    // compute speed, no need for dt since we normalize
    const w = mth.div(mth.neg(mth.add(s, 1.0)), normalizer).map(v=>Math.exp(v));
    return mth.add(mth.mul(w, (maxw-minw)), minw);
}

module.exports = brush;
