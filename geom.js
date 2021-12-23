/**
 *  ______ _______ _____  ___ ___ _______ _______ _______ _______ _______ _______ ______          _____ _______
 * |   __ \       |     ||   |   |     __|       |    |  |     __|       |   |   |   __ \______ _|     |     __|
 * |    __/   -   |       \     /|    |  |   -   |       |__     |   -   |   |   |    __/______|       |__     |
 * |___|  |_______|_______||___| |_______|_______|__|____|_______|_______|_______|___|         |_______|_______|
 *
 * © Daniel Berio (@colormotor) 2021 - ...
 *
 * Geometry utils
 *
 **/

"use strict";

const mth = require("./mth.js");
const alg = require("./alg.js");

const _ = require("lodash");

const geom = function() { };

geom.cleanup_contour = (X, closed = false, eps = 1e-10, get_inds = false) => {
  // Removes points that are closer then a threshold eps
  if (closed)
    X = X.concat([X[0]]);
  var D = mth.diff(X, 0);
  var inds = _.range(0, X.length);
  // chord lengths
  var s = D.map((d) => d[0] ** 2 + d[1] ** 2);

  // Delete values in input with zero distance
  var Y = [X[0]];
  var inds = [0];

  for (var i = 0; i < s.length; i++) {
    if (s[i] <= eps)
      continue;
    Y.push(X[i + 1]);
    inds.push(i + 1);
  }
  if (closed) {
    Y = X.slice(0, Y.length - 1);
    inds = inds.slice(0, inds.length - 1);
  }

  if (get_inds)
    return [Y, inds];
  return Y;
}

// def chord_lengths( P, closed=0 ):
//     ''' Chord lengths for each segment of a contour '''
//     if closed:
//         P = np.vstack([P, P[0]])
//     D = np.diff(P, axis=0)
//     L = np.sqrt( D[:,0]**2 + D[:,1]**2 )
//     return L


// def cum_chord_lengths( P, closed=0 ):
//     ''' Cumulative chord lengths '''
//     if len(P.shape)!=2:
//         return []
//     if P.shape[0] == 1:
//         return np.zeros(1)
//     L = chord_lengths(P, closed)
//     return np.cumsum(np.concatenate([[0.0],L]))

// def chord_length( P, closed=0 ):
//     ''' Chord length of a contour '''
//     if len(P.shape)!=2 or P.shape[0] < 2:
//         return 0.
//     L = chord_lengths(P, closed)
//     return np.sum(L)

geom.schematize = function(P_, C, angle_offset, closed = false, get_edge_inds = false, maxiter = 1000) {
  const cross = (a, b) => {
    let x = a[1] * b[2] - a[2] * b[1];
    let y = a[2] * b[0] - a[0] * b[2];
    let w = a[0] * b[1] - a[1] * b[0];
    return [x, y, w];
  }

  const line_intersection = (s0, s1, eps = 1e-10) => {
    let sp0 = mth.cross([...s0[0], ...[1.0]], [...s0[1], ...[1.0]]);
    let sp1 = mth.cross([...s1[0], ...[1.0]], [...s1[1], ...[1.0]]);
    let ins = mth.cross(sp0, sp1);
    if (Math.abs(ins[2]) < eps)
      return [false, [0, 0]];
    return [true, [ins[0] / ins[2], ins[1] / ins[2]]];
  }

  const project_on_line = (p, a, b) => {
    let d = mth.sub(b, a)
    let t = mth.dot(mth.sub(p, a), d) / mth.dot(d, d);
    return mth.add(a, mth.mul(mth.sub(b, a), t));
  }

  const perp = (v) => {
    return [-v[1], v[0]];
  }

  const cost = (V, n) => {
    let b = mth.mean(V, 0);
    let bn = mth.dot(b, n);
    let c = 0.0;
    for (const v of V) {
      let ct = mth.dot(v, n) - bn;
      c += ct * ct;
    }
    return c
  }

  const best_cost = (V, N) => {
    let costs = N.map((n) => cost(V, n));
    return mth.argmin(costs);
  }

  const merge = (blocks, b0, b1, N) => {
    if (b0.best != b1.best)
      return b1;
    b0.V = [...b0.V, ...b1.V.slice(1)]; // sum(b0.V, b1.V.slice(1)); //[1:]
    b0.best = best_cost(b0.V, N);
    blocks.remove(b1);
    return b0
  }

  let P = [...P_];
  if (closed)
    P.push(P[0]);
  let edges = _.range(0, P.length - 1).map((i) => [i, i + 1]);

  // Orthognonal directions
  let dtheta = Math.PI / 180 * angle_offset;
  let N = []
  for (var i = 0; i < C; i++) {
    var th = (Math.PI / C) * i + dtheta;
    N.push([Math.cos(th), Math.sin(th)]);
  }
  //V = {}
  //best = {}
  let blocks = new alg.DoublyLinkedList();
  for (var i = 0; i < edges.length; i++) {
    let e = edges[i];
    let b = new alg.DoublyLinkedList.Node();
    b.V = [];
    for (var j = 0; j < 2; j++)
      b.V.push(P[e[j]]);
    b.best = best_cost(b.V, N);
    b.edge_index = i;
    blocks.add(b);
  }


  for (var iter = 0; iter < maxiter; iter++) {
    let b0 = blocks.front;
    let b1 = b0.next;
    let num_merges = 0;
    while (b1 != null) {
      if (merge(blocks, b0, b1, N) == b0) {
        num_merges++;
        // backtrack
        b1 = b0;
        if (b1.prev != null) {
          if (merge(blocks, b1.prev, b1, N) == b1.prev) {
            b1 = b1.prev;
            num_merges++;
          }
        }
      }
      b0 = b1
      b1 = b1.next
    }

    if (!num_merges)
      break;
  }

  //return P;

  let Q = [];
  let edge_inds = [];
  let block = blocks.front;
  let prev = null;
  let b_prev = null
  let d_prev = null;
  while (block != null) {
    var b = mth.mean(block.V, 0);
    var n = N[block.best];

    var d = mth.perp(n);
    // block = block.next;
    // continue;

    if (prev != null) {
      const [res, ins] = line_intersection([b_prev, mth.add(b_prev, d_prev)], [b, mth.add(b, d)]);
      if (res) {
        Q.push(ins);
        // if (has_nan(Q[Q.length-1]))
        //   console.log('Nan ins');
        edge_inds.push(block.edge_index);
      }
    } else {
      Q.push(project_on_line(block.V[0], b, mth.add(b, d)));
      // if (has_nan(Q[Q.length-1]))
      //  console.log('Nan proj');
      edge_inds.push(block.edge_index);
    }

    prev = block;
    b_prev = b;
    d_prev = d;
    block = block.next;
  }
  //console.log(Q);
  Q.push(project_on_line(blocks.back.V[blocks.back.V.length - 1], b, mth.add(b, d)));
  // if (has_nan(Q[Q.length-1]))
  //   console.log('Nan end');
  edge_inds.push(blocks.back.edge_index);
  // if (closed && Q.length > 2) {
  //   const [res, ins] = line_intersection([Q[0], Q[1]], [Q[Q.length - 2], Q[Q.length - 3]])
  //   if (res) {
  //     Q[0] = ins;
  //     Q[Q.length - 1] = ins;
  //   } else if (false) { //Q.length) {
  //     Q.pop();
  //     const [res, ins] = line_intersection([Q[0], Q[1]], [Q[Q.length - 2], Q[Q.length - 3]])
  //     if (res) {
  //       Q[0] = ins;
  //       Q[Q.length - 1] = ins;
  //     }
  //   }
  // }
  if (get_edge_inds)
    return [Q, edge_inds];
  return Q;
}

geom.random_radial_polygon = (n, min_r, max_r, center = [0, 0]) => {
  let R = mth.uniform(min_r, max_r, n);
  let start = mth.uniform(0, Math.PI * 2);
  let Theta = mth.randspace(start, start + Math.PI * 2, n + 1).slice(0, -1);
  var res = mth.transpose([mth.add(mth.mul(mth.cos(Theta), R), center[0]),
  mth.add(mth.mul(mth.sin(Theta), R), center[0])]);
  return res;
}

geom.foo = () => 0;

module.exports = geom;
