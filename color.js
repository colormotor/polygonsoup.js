"use strict"

const color = function () {}

color.categorical = {
    default12 : [[255, 109, 174],
  [212, 202, 58],
  [0, 190, 255],
  [235, 172, 250],
  [158, 158, 158],
  [103, 225, 181],
  [214, 39, 40],
  [154, 208, 255],
  [226, 133, 68],
  [31, 119, 180],
  [255, 127, 14],
  [93, 177, 90]]

};

color.current = color.categorical.default12;
color.default = (i) => color.current[i%color.current.length];

module.exports = color;
