/**
 *  ______ _______ _____  ___ ___ _______ _______ _______ _______ _______ _______ ______          _____ _______
 * |   __ \       |     ||   |   |     __|       |    |  |     __|       |   |   |   __ \______ _|     |     __|
 * |    __/   -   |       \     /|    |  |   -   |       |__     |   -   |   |   |    __/______|       |__     |
 * |___|  |_______|_______||___| |_______|_______|__|____|_______|_______|_______|___|         |_______|_______|
 *
 * © Daniel Berio (@colormotor) 2021 - ...
 *
 * wrappper around p5js
 *
 **/

'use strict';
var p5 = require("p5");
var mth = require("./mth.js");

p5.prototype.polygon2 = function(P, closed=false){
  this.beginShape();
  for (const p of P)
    this.vertex(p[0], p[1]);
  if (closed)
    this.endShape(this.CLOSE);
  else
    this.endShape();
}

p5.prototype.plot = function(rect, y){
  this.beginShape();

  const w = rect[1][0] - rect[0][0];
  const h = rect[1][1] - rect[0][1];
  y = mth.div(y, mth.max(y));
  const x = mth.linspace(rect[0][0], rect[1][0], y.length);
  this.vertex(x[0], rect[1][1]);
  for (var i = 0; i < y.length; i++)
    this.vertex(x[i], rect[1][1] - y[i]*h);
  this.vertex(rect[1][0], rect[1][1]);
  this.endShape(this.CLOSE);
}



module.exports = p5;
