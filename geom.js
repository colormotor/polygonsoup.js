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

/// Removes contour points that are closer then eps
geom.cleanup_contour = (X, closed = false, eps = 1e-10, get_inds = false) => {
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

/// Chord lengths for each segment of a contour
geom.chord_lengths = ( P, closed=0 ) => {
  if (closed)
    P = P.concat([P[0]]); //[...P, ...[P[0]]];
  var D = mth.diff(P, 0);
  return D.map((d) => d[0] ** 2 + d[1] ** 2);
}

/// Cumulative chord lengths
geom.cum_chord_lengths = ( P, closed=0 ) => {
  if (P.length < 2)
    return [];
  var L = geom.chord_lengths(P, closed);
  return mth.cumsum([0.0].concat(L));
}

geom.chord_length = ( P, closed=0 ) => {
  if (P.length < 2)
    return 0;
  var L = geom.chord_lengths(P, closed);
  return mth.sum(L);
}

geom.is_compound = (S) => {
  var d = mth.dim(S);
  if (d > 3)
    return true;
  return false;
}

const affine_transform = (mat, data) => {
  if (mth.dim(data)==2)
    return mth.slice( mth.dot(mat, data.concat([1.0])), -1 );
  if (!geom.is_compound(data)){
    var n = data.length;
    return mth.transpose(
      mth.slice( mth.dot(mat, mth.vstack(mth.transpose(data), mth.ones(n))), -1));
  }
  // compound
  return data.map(P=>affine_transform(mat, P));
}
geom.affine_transform = affine_transform;
geom.affine = affine_transform;

geom.bounding_box = (S, padding=0) => {
    /// Axis ligned bounding box of one or more contours (any dimension)
    // Returns [min,max] list'''
  if (!S.length)
    return [[0,0], [0,0]];

  if (!is_compound(S))
    S = [S];

  bmin = mth.min(S.map(V=>mth.min(V, 0)), 0);
  bmin = mth.max(S.map(V=>mth.max(V, 0)), 0);
  return [mth.sub(bmin, padding), mth.add(bmax, padding)]
}

geom.make_rect = (x, y, w, h) => {
    return [[x, y], [x+w, y+h]];
}

geom.make_centered_rect = (p, size) => {
  return geom.make_rect(p[0] - size[0]*0.5, p[1] - size[1]*0.5, size[0], size[1]);
}

geom.pad_rect = (rect, pad) => {
  return [mth.add(rect[0], pad), mth.sub(rect[1], pad)];
}

geom.rect_size = (rect) => mth.sub(rect[1] - rect[0]);
geom.rect_aspect = (rect) => mth.sub(rect[0] - rect[1]);
geom.rect_w = (rect) => geom.rect_size(rect)[0];
geom.rect_h = (rect) => geom.rect_size(rect)[0];
geom.rect_center = (rect) => mth.add(rect[0], mth.div(mth.sub(rect[1], rect[0]), 2));
geom.rect_l = (rect) => rect[0][0];
geom.rect_r = (rect) => rect[1][0];
geom.rect_t = (rect) => rect[0][1];
geom.rect_b = (rect) => rect[1][1];

geom.rect_corners = (rect, close=false) => {
  var [w, h] = rect_size(rect);
  var P = [mth.add(rect[0], 0), mth.add(rect[0], [w, 0]),
       mth.add(rect[1], 0), mth.add(rect[0] + [0, h])];
  if (close)
    P.append(P[0]);
  return P
}

geom.random_point_in_rect = (rect) => {
  var x = mth.uniform( rect[0][0], rect[1][0] );
  var y = mth.uniform( rect[0][1], rect[1][1] );
  return [x, y];
}

geom.scale_rect = (rect, s, halign=0, valign=0) => {
  if (mth.is_number(s))
    s = [s, s];
  var [sx, sy] = s;
  var r = rect.map(p=>p.slice(0));
  var origin = geom.rect_center(rect);
  if (halign == -1)
    origin[0] = geom.rect_l(rect);
  if (halign == 1)
    origin[0] = geom.rect_r(rect)
  if (valign == -1)
    origin[1] = geom.rect_t(rect);
  if (valign == 1)
    origin[1] = geom.rect_b(rect);
  var A = mth.matmul(geon.trans_2d(origin), geom.scaling_2d([sx, sy]), mth.neg(geom.trans_2d(origin)));
  return [geom.affine_transform(A, r[0]), geom.affine_transform(A, r[1])];
}

/// Fit src rect into dst rect, preserving aspect ratio of src, with optional padding
geom.rect_in_rect = (src, dst, padding=0., axis=-1) => {
  var dst = geom.pad_rect(dst, padding);
  var [dst_w, dst_h] = mth.sub(dst[1], dst[0]);
  var [src_w, src_h] = mth.sub(src[1], src[0]);

  var ratiow = dst_w/src_w;
  var ratioh = dst_h/src_h;

  if (axis < 0){
    if (ratiow <= ratioh)
        axis = 1;
    else
        axis = 0;
  }
  if (axis==1){  // fit vertically [==]
    var w = dst_w;
    var h = src_h*ratiow;
    var x = dst[0][0];
    var y = dst[0][1] + dst_h*0.5 - h*0.5;
  }else{        // fit horizontally [ || ]
    var w = src_w*ratioh;
    var h = dst_h;
    var y = dst[0][1];
    var x = dst[0][0] + dst_w*0.5 - w*0.5;
  }
  return geom.make_rect(x, y, w, h);
}

/// Return homogeneous transformation matrix that fits src rect into dst
geom.rect_in_rect_transform = (src, dst, padding=0., axis=None) => {
  var fitted = geom.rect_in_rect(src, dst, padding, axis);
  var cenp_src = geom.rect_center(src);
  var cenp_dst = geom.rect_center(fitted);

  var M = mth.eye(3);
  M = mth.dot(M,
              geom.trans_2d(mth.sub(cenp_dst, cenp_src)));
  M = mth.dot(M, geom.trans_2d(cenp_src));
  M = mth.dot(M, geom.scaling_2d(mth.div(rect_size(fitted), rect_size(src))));
  M = mth.dot(M, geom.trans_2d(mth.neg(cenp_src)));
  return M;
}

// Intersections and all
geom.line_intersection = (s0, s1, eps = 1e-10) => {
  let sp0 = mth.cross([...s0[0], ...[1.0]], [...s0[1], ...[1.0]]);
  let sp1 = mth.cross([...s1[0], ...[1.0]], [...s1[1], ...[1.0]]);
  let ins = mth.cross(sp0, sp1);
  if (Math.abs(ins[2]) < eps)
    return [false, [0, 0]];
  return [true, [ins[0] / ins[2], ins[1] / ins[2]]];
}

geom.project_on_line = (p, a, b) => {
    let d = mth.sub(b, a)
    let t = mth.dot(mth.sub(p, a), d) / mth.dot(d, d);
    return mth.add(a, mth.mul(mth.sub(b, a), t));
}


geom.line_intersection_uv = ( a1, a2, b1, b2, aIsSegment=False, bIsSegment=False) => {
  const EPS = 0.00001;
  var intersection = mth.zeros(2);
  var uv = mth.zeros(2)

  const denom  = (b2[1]-b1[1]) * (a2[0]-a1[0]) - (b2[0]-b1[0]) * (a2[1]-a1[1]);
  const numera = (b2[0]-b1[0]) * (a1[1]-b1[1]) - (b2[1]-b1[1]) * (a1[0]-b1[0]);
  const numerb = (a2[0]-a1[0]) * (a1[1]-b1[1]) - (a2[1]-a1[1]) * (a1[0]-b1[0]);

  if (Math.abs(denom) < EPS)
    return [false, intersection, uv];

  uv[0] = numera / denom;
  uv[1] = numerb / denom;

  intersection[0] = a1[0] + uv[0] * (a2[0] - a1[0]);
  intersection[1] = a1[1] + uv[0] * (a2[1] - a1[1]);

  var isa = true;
  if (aIsSegment && (uv[0]  < 0 || uv[0]  > 1))
    isa = false;
  var isb = true;
  if (bIsSegment && (uv[1] < 0 || uv[1]  > 1))
    isb = false;

  return [isa&&isb, intersection, uv];
}

geom.line_segment_intersection = ( a1, a2, b1, b2 ) => {
  var [res, intersection, uv] = geom.line_intersection_uv(a1,a2,b1,b2,false,true);
    return [res, intersection];
}

geom.segment_line_intersection = ( a1, a2, b1, b2 ) => {
    var [res, intersection, uv] = geom.line_intersection_uv(a1,a2,b1,b2,true,false);
    return [res, intersection];
}

geom.segment_intersection = ( a1, a2, b1, b2 )  => {
    var [res, intersection, uv] = geom.line_intersection_uv(a1,a2,b1,b2,true,true);
    return [res, intersection];
}

geom.line_ray_intersection = ( a1, a2, b1, b2 )  => {
    var [res, intersection, uv] = geom.line_intersection_uv(a1,a2,b1,b2,false,false);
    return [res && uv[1] > 0, intersection];
}

geom.ray_line_intersection = ( a1, a2, b1, b2 )  => {
    var [res, intersection, uv] = geom.line_intersection_uv(a1,a2,b1,b2,false,false);
    return [res && uv[0] > 0, intersection];
}

geom.ray_intersection = ( a1, a2, b1, b2 )  => {
    var [res, intersection, uv] = geom.line_intersection_uv(a1,a2,b1,b2,false,false);
    return [res && uv[0] > 0 && uv[1] > 0, intersection];
}

geom.ray_segment_intersection = ( a1, a2, b1, b2 )  => {
    var [res, intersection, uv] = geom.line_intersection_uv(a1,a2,b1,b2,false,true);
    return [res && uv[0] > 0 && uv[1] > 0, intersection];
}




geom.random_radial_polygon = (n, min_r, max_r, center = [0, 0]) => {
  let R = mth.uniform(min_r, max_r, n);
  let start = mth.uniform(0, Math.PI * 2);
  let Theta = mth.randspace(start, start + Math.PI * 2, n + 1).slice(0, -1);
  var res = mth.transpose([mth.add(mth.mul(mth.cos(Theta), R), center[0]),
                           mth.add(mth.mul(mth.sin(Theta), R), center[1])]);
  return res;
}


/**
 * schematize - C-oriented polyline schematization.
 * Quick and dirty implementation of angular schematization as in:
 * Dwyer et al. (2008) A fast and simple heuristic for metro map path simplification
 * https://pdfs.semanticscholar.org/c791/260d13dff6a7fad539ae182e9791c909aa17.pdf
 * still can produce some very short segments, must fix.
 **/
geom.schematize = (P_, C, angle_offset, closed = false, get_edge_inds = false, maxiter = 1000) => {
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
    b0.V = [...b0.V, ...b1.V.slice(1)];
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
      const [res, ins] = geom.line_intersection([b_prev, mth.add(b_prev, d_prev)], [b, mth.add(b, d)]);
      if (res) {
        Q.push(ins);
        edge_inds.push(block.edge_index);
      }
    } else {
      Q.push(geom.project_on_line(block.V[0], b, mth.add(b, d)));
      edge_inds.push(block.edge_index);
    }

    prev = block;
    b_prev = b;
    d_prev = d;
    block = block.next;
  }

  Q.push(geom.project_on_line(blocks.back.V[blocks.back.V.length - 1], b, mth.add(b, d)));
  edge_inds.push(blocks.back.edge_index);
  // FIXME
  // if (closed && Q.length > 2) {
  //   const [res, ins] = line_intersection([Q[0], Q[1]], [Q[Q.length - 2], Q[Q.length - 3]])
  //   if (res) {
  //     Q[0] = ins;
  //     Q[Q.length - 1] = ins;
  //   } else if  (false) { //Q.length) {
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

geom.foo = () => 0;

module.exports = geom;
