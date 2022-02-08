/**
 *  ______ _______ _____  ___ ___ _______ _______ _______ _______ _______ _______ ______          _____ _______
 * |   __ \       |     ||   |   |     __|       |    |  |     __|       |   |   |   __ \______ _|     |     __|
 * |    __/   -   |       \     /|    |  |   -   |       |__     |   -   |   |   |    __/______|       |__     |
 * |___|  |_______|_______||___| |_______|_______|__|____|_______|_______|_______|___|         |_______|_______|
 *
 * © Daniel Berio (@colormotor) 2021 - ...
 *
 * Math utilities
 * Currently wraps numericjs (https://ccc-js.github.io/numeric2/documentation.html)
 * Maybe there is a better linalg lib?
 **/

'use strict';
var mth = require("numericjs");
const _ = require("lodash");
const seedrandom = require('seedrandom');
const mvn = require("multivariate-normal").default;

var rng = seedrandom();

mth.random_seed = (v) => {
  rng = seedrandom(v);
  mth.rand = rng;
}

mth.rand = rng; //Math.random
mth.eye = mth.identity;
mth.norm = mth.norm2;

mth.zero_eps = 1e-15;

/* Von Mises distribution centered at mu and with spread kappa */
mth.von_mises_like = ( theta, kappa, mu=0.0 ) => {
  return Math.exp( kappa * Math.cos(theta-mu) );
}

/* From https://hewgill.com/picomath/javascript/erf.js.html*/
mth.erf = (x) => {
    // constants
    var a1 =  0.254829592;
    var a2 = -0.284496736;
    var a3 =  1.421413741;
    var a4 = -1.453152027;
    var a5 =  1.061405429;
    var p  =  0.3275911;

    // Save the sign of x
    var sign = 1;
    if (x < 0) {
        sign = -1;
    }
    x = Math.abs(x);

    // A&S formula 7.1.26
    var t = 1.0/(1.0 + p*x);
    var y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*Math.exp(-x*x);
    return sign*y;
}


mth.gaussian = (mean, stddev, randf=mth.rand) => {
    return function() {
        var V1
        var V2
        var S
        do{
          var U1 = randf;
          var U2 = randf;
            V1 = 2*U1-1
            V2 = 2*U2-1
            S = V1*V1+V2*V2
        }while(S >= 1)
        if(S===0) return 0
        return mean+stddev*(V1*Math.sqrt(-2*Math.log(S)/S))
    }
}

mth.multivariate_gaussian = (mean, cov) => {
  let distribution = mvn(mean, cov);
  return distribution.sample;
}


mth.randint = (min, max) => Math.floor(mth.rand() * (max - min) ) + min;
mth.random_choice = (ar) => ar[mth.randint(0, ar.length)];

mth.clamp = (num, min, max) => Math.min(Math.max(num, min), max);
mth.clip = mth.clamp

mth.compare_nums = (a, b) => {
  return a - b;
}

mth.uniform = (a = 0.0, b = 1.0, n = 1) => {
  if (n == 1)
    return a + mth.rand() * (b - a);
  let v = new ArrayT(n);
  for (var i = 0; i < n; i++)
    v[i] = mth.uniform(a, b);
  return v;
}

mth.shuffled = (array) => {
  array = array.slice(0);
    for (let i = array.length - 1; i > 0; i--) {
        let j = Math.floor(mth.rand() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
  return array;
}

const gauss1 = mth.gaussian(0, 1);

mth.random = {
  seed: (n) => mth.random_seed(n),
  uniform: (a = 0.0, b = 1.0, n = 1) => mth.uniform(a, b, n),
  normal: (range=1, n=1) => {
    if (n==1)
      return gauss1()*range;
    return _.range(0, n).map(i=>gauss1()*range);
  },
  choice: (ar) => mth.random_choice(ar)
};

const ArrayT = Array; // Float64Array;
// Rare docs for numericjs
// https://ccc-js.github.io/numeric2/documentation.html

mth.is_number = (v) => typeof(v) == 'number';

mth.matmul = function() {
  var m = arguments[0];
  for (var i = 1; i < arguments.length; i++)
    m = mth.dot(m, arguments[i]);
  return m;
}

mth.slice = function() {
  if (arguments.length==1)
    return arguments[0];
  if (arguments.length==2){
    var [v, to] = arguments;
    var from = 0;
  }else if(arguments.length==3){
    var [v, from, to] = arguments;
  }

  if (from < 0)
    from = v.length + from;
  if (to < 0)
    to = v.length + to;
  return v.slice(from, to);
}

mth.neg = (v) => mth.mul(v, -1);

mth.eyev = (n, v) => {
  var res = mth.zeros(n, n);
  for (var i = 0; i < n; i++)
    res[i][i] = v;
  return res;
}

mth.shiftmat = (n, o) => {
  var m = mth.zeros(n, n);
  for (var i = 0; i < n; i++)
    if (i + o >= 0 && i + o < n)
      m[i][i + o] = 1;
  return m;
}

const ones = (nrow, ncol = 0) => {
  if (ncol == 0) {
    var x = new ArrayT(nrow);
    for (var i = 0; i < nrow; i++)
      x[i] = 1;
    return x;
  }

  return _.range(0, nrow).map((i) => ones(ncol, 0));
}
mth.ones = ones

const zeros = (nrow, ncol = 0) => {
  if (ncol == 0) {
    var x = new ArrayT(nrow);
    for (var i = 0; i < nrow; i++)
      x[i] = 0;
    return x;
  }

  return _.range(0, nrow).map((i) => mth.zeros(ncol, 0));
}
mth.zeros = zeros

mth.concatenate = (ars) => {
  var n = mth.sum(ars.map((a) => a.length));
  var ar = new ArrayT(n);
  var i = 0;
  for (const a of ars) {
    //ar.set(i, a);
    for (var j = 0; j < a.length; j++)
      ar[i + j] = a[j];
    i += a.length;
  }
  return ar;
}

mth.vstack = (ars) => {
  return Array.prototype.concat(...ars);
}

mth.hstack = (ars) => {
  return ars[0].map((row, i) => mth.concatenate(ars.map((ar) => ar[i])));
}

mth.submat = (mat, rows, cols) => {
  if (!rows.length)
    rows = [0, mat.length];
  if (!cols.length)
    cols = [0, mat[0].length];

  var res = [];
  for (var i = rows[0]; i < rows[1]; i++) {
    res.push(_.range(cols[0], cols[1]).map((j) => mat[i][j]));
  }
  return res;
}

mth.set_submat = (mat, rows, cols, block) => {
  if (!rows.length)
    rows = [0, mat.length];
  if (!cols.length)
    cols = [0, mat[0].length];

  for (var i = rows[0]; i < rows[1]; i++) {
    for (var j = cols[0]; j < cols[1]; j++)
      mat[i][j] = block[i - rows[0]][j - cols[0]];
  }
  return mat;
}

mth.block = (ars) => {
  return mth.vstack(ars.map((row) => mth.hstack(row)));
}

mth.block_diag = (ars) => {
  var [rows, cols] = mth.dim(ars[0]);
  var n = ars.length;
  var res = mth.zeros(rows * n, cols * n);
  for (var i = 0; i < n; i++)
    mth.set_submat(res, [i * rows, i * rows + rows], [i * cols, i * cols + cols], ars[i]);
  return res;
}

mth.repmat = (mat, rows, cols) => {
  if (mth.dim(mat).length < 2)
    mat = [mat];
  var [n, m] = mth.dim(mat);
  var res = mth.zeros(n * rows, m * cols);
  for (var i = 0; i < rows; i++) {
    for (var j = 0; j < cols; j++) {
      mth.set_submat(res, [i * n, i * n + n], [j * m, j * m + m], mat);
    }
  }
  return res;
}

mth.reshape = (x, shape) => {
  var k = 0;
  var y = [];
  if (mth.dim(x).length > 1)
    x = mth.concatenate(x);
  for (var i = 0; i < shape[0]; i++) {
    var row = [];
    for (var j = 0; j < shape[1]; j++)
      row.push(x[k++]);
    y.push(row);
  }
  return y;
}

const has_nan = (v) => {

  if (mth.dim(v).length < 2){
    for (const x of v)
      if (isNaN(x))
        return true;
    return false;
  }
  for (const row of v)
    if (has_nan(row))
      return true;
  return false;
};
mth.has_nan = has_nan;

const has_undefined = (v) => {
  if (mth.dim(v).length < 2)
    for (const x of v)
      if (x==undefined)
        return true;
  return false;

  for (const row of v)
    if (has_undefined(row))
      return true;

  return false;
}
mth.has_undefined = has_undefined;

mth.print = (v, txt = "") => {
  console.log(txt);
  if (typeof (v) == 'number') {
    console.log(v);
    return;
  }

  if (mth.dim(v).length < 2) {
    console.log("[" + v.toString() + "]");
    return;
  }

  var s = "[";
  for (var i = 0; i < v.length; i++) {
    if (i != 0)
      s = s + " ";
    s = s + "[" + v[i].toString() + "]";
    if (i < v.length - 1)
      s = s + "\n";
  }
  s += "]";
  console.log(s);
}


const sum1 = mth.sum;
mth.sum = (x, axis = -1) => {
  const shape = mth.dim(x);
  if (shape.length == 1 || axis < 0)
    return sum1(x);
  if (axis == 1)
    return x.map((row) => sum1(row));
  var res = mth.zeros(shape[1]);
  for (var i = 0; i < shape[1]; i++)
    for (var j = 0; j < shape[0]; j++)
      res[i] += x[j][i];
  return res;
}


mth.cumsum = (v) => {
  let c = new ArrayT(v.length);
  let s = 0;
  for (var i = 0; i < v.length; i++) {
    c[i] = s + v[i];
    s += v[i];
  }
  return c;
}

const diff1 = (x) => x.slice(1).map((item, index) => { return item - x[index] });

mth.diff = (x, axis = 0) => {
  const shape = mth.dim(x);
  if (shape.length == 1)
    return diff1(x);
  if (axis == 1)
    return x.map((row) => diff1(row));
  // axis == 0
  return mth.transpose(mth.diff(mth.transpose(x), 1));
}

mth.delete_elements = (x, I) => {
  var y = x.slice(0);
  y.splice(0, I.length, ...I);
  return y;
}

mth.mean = (v, axis = 0) => {
  const shape = mth.dim(v);
  if (shape.length == 1)
    return mth.sum(v) / v.length;
  if (axis == 1)
    return v.map((row) => mth.sum(row) / row.length);
  else
    return mth.transpose(v).map((row) => mth.sum(row) / row.length);
}

mth.cross = (a, b) => {
  let x = a[1] * b[2] - a[2] * b[1];
  let y = a[2] * b[0] - a[0] * b[2];
  let z = a[0] * b[1] - a[1] * b[0];
  return [x, y, z];
}

mth.perp = (v) => {
  return [-v[1], v[0]];
}

mth.radians = (x) => Math.PI / 180 * x;

mth.degrees = (x) => x * (180.0 / Math.PI);

mth.normalize = (v) => mth.div(v, mth.norm(v));

mth.angle_between = (a, b) => {
  return Math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]);
}

mth.distance = (a, b) => mth.norm(mth.sub(b, a));
mth.distance_sq = (a, b) => mth.dot(mth.sub(b, a), mth.sub(b, a));

// https://gist.github.com/janosh/099bd8061f15e3fbfcc19be0e6b670b9
mth.argfact = (compareFn) => (array) => array.map((el, idx) => [el, idx]).reduce(compareFn)[1]
mth.argmax = mth.argfact((min, el) => (el[0] > min[0] ? el : min))
mth.argmin = mth.argfact((max, el) => (el[0] < max[0] ? el : max))

mth.imod = (n, m) => {
  return ((n % m) + m) % m;
}

mth.argsort = (x) => {
  let I = _.range(0, x.length);
  I.sort((i,j)=>mth.compare_nums(x[i], x[j]));
  return I;
}

// mth.factorial = (num) => {
//     var v=1;
//     for (var i = 2; i <= num; i++)
//         v = v * i;
//     return v;
// }

const factorial = n => !(n > 1) ? 1 : factorial(n - 1) * n;
mth.factorial = factorial

mth.rotate_array = function(ar, n) {
  var n = mth.imod(n, ar.length);
  return ar.slice(n, ar.length).concat(ar.slice(0, n));
}

const min = (ar, axis=-1) => {
  if (mth.dim(ar).length > 1) {
    if (axis==0)
      return min(mth.transpose(ar), axis=1);
    else if (axis==1)
      return ar.map(row=>Math.min.apply(null, row));
    else
      return Math.min.apply(null, ar.map(row=>Math.min.apply(null, row)));
  }
  return Math.min.apply(null, ar);
}
mth.min = min;

const max = (ar, axis=-1) => {
  if (mth.dim(ar).length > 1) {
    if (axis==0)
      return max(mth.transpose(ar), axis=1);
    else if (axis==1)
      return ar.map(row=>Math.max.apply(null, row));
    else
      return Math.max.apply(null, ar.map(row=>Math.max.apply(null, row)));
  }
  return Math.max.apply(null, ar);
}
mth.max = max;

mth.randspace = (a, b, n, minstep = 0.1, maxstep = 0.5) => {
  var v = mth.uniform(minstep, maxstep, n);
  v[0] = 0;
  v = mth.div(v, mth.sum(v));
  v = mth.cumsum(v);
  return mth.add(mth.mul(v, b - a), a);
}

mth.lerp = (a, b, t) => {
  return mth.add(a, mth.mul(mth.sub(b, a), t));
}

mth.madd = (a, b, c) => {
  return mth.add(mth.mul(a, b), c);
}

mth.addm = (a, b, c) => {
  return mth.add(a, mth.mul(b, c));
}

mth.msub = (a, b, c) => {
  return mth.sub(mth.mul(a, b), c);
}

mth.subm = (a, b, c) => {
  return mth.sub(a, mth.mul(b, c));
}

mth.quantize = (v, step) => Math.round(v / step) * step;

mth.dist2sq = (a, b) => (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]);

const min_eps = 0.1;
mth.min3 = (a, b, c) => {
  // if (Math.abs(c - a) < min_eps && Math.abs(c - b) < min_eps)
  //   return c;

  var m = a;
  if (b <= m)
    m = b;
  if (c <= m)
    m = c;
  return m;
}

mth.argmin3 = (a, b, c) => {
  // if (Math.abs(c - a) < min_eps && Math.abs(c - b) < min_eps)
  //   return 2;

  var m = a;
  var i = 0;
  if (b <= m) {
    m = b;
    i = 1;
  }
  if (c <= m)
    return 2;
  return i;
}

/// Kronecker product
/// from mathjs https://github.com/josdejong/mathjs
mth.kron = (a, b) => {
  // Deal with the dimensions of the matricies.
  if (mth.dim(a).length === 1) {
    // Wrap it in a 2D Matrix
    a = [a]
  }
  if (mth.dim(b).length === 1) {
    // Wrap it in a 2D Matrix
    b = [b]
  }
  if (mth.dim(a).length > 2 || mth.dim(b).length > 2) {
    throw new RangeError('Vectors with dimensions greater then 2 are not supported expected ' +
      '(Size x = ' + JSON.stringify(a.length) + ', y = ' + JSON.stringify(b.length) + ')')
  }

  const t = []
  let r = []

  return a.map(function(a) {
    return b.map(function(b) {
      r = []
      t.push(r)
      return a.map(function(y) {
        return b.map(function(x) {
          return r.push(mth.mul(y, x))
        })
      })
    })
  }) && t
}

/**
 * Dynamic time warping (DTW) between two sequences
 * @param {any} x first sequence
 * @param {any} y second sequrence
 * @param {any} w maximum window size
 * @param {any} get_dist if true get DTW get distance
 * @param {any} distfn distance function
 * @returns returns the optimal path between the sequences and optionallly the distance
 */
mth.dtw = (x, y, w = Infinity, get_dist = false, distfn = mth.dist2sq) => {
  const nx = x.length;
  const ny = y.length;
  w = Math.max(w, Math.abs(nx - ny));

  var D = _.range(0, nx).map((r) => _.range(0, ny).map((c) => Infinity));//  mth.mul(mth.ones(nx + 1, ny + 1), Infinity);
  D[0][0] = 0;

  for (var i = 1; i < nx; i++)
    for (var j = Math.max(1, i - w); j < Math.min(ny, i + w); j++)
      D[i][j] = 0;

  for (var i = 1; i < nx; i++)
    for (var j = Math.max(1, i - w); j < Math.min(ny, i + w); j++)
      D[i][j] = distfn(x[i], y[j]) + mth.min3(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1]);


  if (get_dist)
    return D[nx - 1][ny - 1];

  var i = nx - 1;
  var j = ny - 1;
  var p = []

  // recover path
  p.push([i, j]);
  while (i > 0 && j > 0) {
    //var id = mth.argmin([D[i][j-1], D[i-1][j], D[i-1][j-1]]);
    var id = mth.argmin3(D[i][j - 1], D[i - 1][j], D[i - 1][j - 1]);
    if (id == 0) {
      j = j - 1;
    } else if (id == 1) {
      i = i - 1;
    } else {
      i = i - 1;
      j = j - 1;
    }
    p.push([i, j]);
  }

  return p.reverse();
}

mth.dtw_path = (x, y, w = Infinity) => {
  return mth.dtw(x, y, w, false);
}

mth.dtw_dist = (x, y, w = Infinity) => {
  return mth.dtw(x, y, w, true);
}

mth.dtw_path_all = (x, y, w = Infinity) => {
  var dists = [];
  var ny = y.length;
  var mind = Infinity;

  for (var i = 0; i < ny-1; i++)
    dists.push(mth.dtw_dist(x, mth.rotate_array(y, i), w));
  var start_index = mth.argmin(dists);
  var path = mth.dtw_path(x, mth.rotate_array(y, start_index), w);
  path = path.map((r) => [r[0], (r[1] + start_index)%ny]);
  return path;
}

module.exports = mth;
