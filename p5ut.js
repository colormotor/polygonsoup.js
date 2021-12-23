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

p5.prototype.polygon2 = function(P, closed=false){
  this.beginShape();
  for (const p of P)
    this.vertex(p[0], p[1]);
  if (closed)
    this.endShape(p5.CLOSE);
  else
    this.endShape();
}

module.exports = p5;
