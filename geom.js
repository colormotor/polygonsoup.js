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

/// Chord lengths for each segment of a contour
geom.chord_lengths = (P, closed = 0) => {
  if (closed)
    P = P.concat([P[0]]); //[...P, ...[P[0]]];
  var D = mth.diff(P, 0);
  return D.map((d) => d[0] ** 2 + d[1] ** 2);
}

/// Cumulative chord lengths
geom.cum_chord_lengths = (P, closed = 0) => {
  if (P.length < 2)
    return [];
  var L = geom.chord_lengths(P, closed);
  return mth.cumsum([0.0].concat(L));
}

geom.chord_length = (P, closed = 0) => {
  if (P.length < 2)
    return 0;
  var L = geom.chord_lengths(P, closed);
  return mth.sum(L);
}

geom.is_compound = (S) => {
  var d = mth.dim(S).length;
  if (d > 2)
    return true;
  return false;
}

geom.edges = (P, closed=false) => {
  const n = P.length;
  let m = n;
  if (!closed)
    m = m-1;
  return _.range(0, m).map((i) => [i, (i + 1)%n]);
}

geom.segments = (S, closed=false) => {
  if (!geom.is_compound(S))
    S = [S];
  let segs = [];
  for (const P of S){
    let n = P.length;
    let m = n;
    if (!closed)
      m = m-1;
    for (let i = 0; i < m; i++){
      segs.push([P[i], P[(i+1)%n]]);
    }
  }
  return segs;
}

geom.point_line_distance = (p, a, b) => {
  if (mth.distance(a, b) < mth.zero_eps)  // check for degenerate line
    return mth.distance(a, p);
  return Math.abs(mth.det([mth.sub(b, a), mth.sub(a, p)])) / mth.distance(b, a);
}

geom.point_segment_distance = (p, a, b) => {
  const d = mth.sub(b, a);
  const u = mth.clamp(mth.dot(mth.sub(p, a), d) / mth.dot(d, d), 0, 1);
  const proj = mth.add(a, mth.mul(d, u));
  return mth.distance(proj, p);
}

geom.project_on_line = (p, a, b) => {
  let d = mth.sub(b, a)
  let t = mth.dot(mth.sub(p, a), d) / mth.dot(d, d);
  return mth.add(a, mth.mul(mth.sub(b, a), t));
}

geom.rot_2d = (theta, affine = true) => {
  let d = affine?3:2;

  let m = mth.eye(d);
  let ct = Math.cos(theta);
  let st = Math.sin(theta);
  m[0][0] = ct; m[0][1] = -st;
  m[1][0] = st; m[1][1] = ct;
  return m;
}

geom.trans_2d = (xy) => {
  var m = mth.eye(3);
  m[0][2] = xy[0]
  m[1][2] = xy[1]
  return m
}

geom.scaling_2d = (xy, affine = true) => {
  if (affine)
    var d = 3;
  else
    var d = 2;

  if (mth.is_number(xy))
    xy = [xy, xy]

  var m = mth.eye(d)
  m[0][0] = xy[0]
  m[1][1] = xy[1]
  return m
}

geom.shear_2d = (xy, affine = true) => {
  if (affine)
    var d = 3;
  else
    var d = 2;

  var m = mth.eye(d)
  m[0][1] = xy[0]
  m[1][0] = xy[1]
  return m
}

const affine_transform = (mat, data) => {
  if (mth.dim(data).length == 1) {
    return mth.slice(mth.dot(mat, data.concat([1.0])), -1);
  }
  if (!geom.is_compound(data)) {
    var n = data.length;
    var res = mth.transpose(
      mth.slice(mth.dot(mat, mth.vstack([mth.transpose(data), [mth.ones(n)]])), -1));
    return res;
  }
  // compound
  return data.map(P => affine_transform(mat, P));
}
geom.affine_transform = affine_transform;
geom.affine = affine_transform;

geom.bounding_box = (S, padding = 0) => {
  /// Axis ligned bounding box of one or more contours (any dimension)
  // Returns [min,max] list'''
  if (!S.length)
    return [[0, 0], [0, 0]];

  if (!geom.is_compound(S))
    S = [S];

  var bmin = mth.min(S.map(V => mth.min(V, 0)), 0);
  var bmax = mth.max(S.map(V => mth.max(V, 0)), 0);
  return [mth.sub(bmin, padding), mth.add(bmax, padding)]
}

geom.make_rect = (x, y, w, h) => {
  return [[x, y], [x + w, y + h]];
}

geom.make_centered_rect = (p, size) => {
  return geom.make_rect(p[0] - size[0] * 0.5, p[1] - size[1] * 0.5, size[0], size[1]);
}

geom.pad_rect = (rect, pad) => {
  return [mth.add(rect[0], pad), mth.sub(rect[1], pad)];
}

geom.rect_size = (rect) => mth.sub(rect[1], rect[0]);
geom.rect_aspect = (rect) => mth.div(rect[0], rect[1]);
geom.rect_w = (rect) => geom.rect_size(rect)[0];
geom.rect_h = (rect) => geom.rect_size(rect)[0];
geom.rect_center = (rect) => mth.add(rect[0], mth.div(mth.sub(rect[1], rect[0]), 2));
geom.rect_l = (rect) => rect[0][0];
geom.rect_r = (rect) => rect[1][0];
geom.rect_t = (rect) => rect[0][1];
geom.rect_b = (rect) => rect[1][1];

geom.rect_corners = (rect, close = false) => {
  let [w, h] = geom.rect_size(rect);
  let P = [mth.add(rect[0], [0,0]), mth.add(rect[0], [w, 0]),
           mth.add(rect[0], [w,h]), mth.add(rect[0], [0, h])];
  //if (close)
  //  P.push(P[0]);
  return P;
}

geom.random_point_in_rect = (rect) => {
  var x = mth.uniform(rect[0][0], rect[1][0]);
  var y = mth.uniform(rect[0][1], rect[1][1]);
  return [x, y];
}

geom.scale_rect = (rect, s, halign = 0, valign = 0) => {
  if (mth.is_number(s))
    s = [s, s];
  var [sx, sy] = s;
  var r = rect.map(p => p.slice(0));
  var origin = geom.rect_center(rect);
  if (halign == -1)
    origin[0] = geom.rect_l(rect);
  if (halign == 1)
    origin[0] = geom.rect_r(rect)
  if (valign == -1)
    origin[1] = geom.rect_t(rect);
  if (valign == 1)
    origin[1] = geom.rect_b(rect);

  var A = mth.matmul(geom.trans_2d(origin), geom.scaling_2d([sx, sy]), geom.trans_2d(mth.neg(origin)));
  return [geom.affine_transform(A, r[0]), geom.affine_transform(A, r[1])];
}

/// Fit src rect into dst rect, preserving aspect ratio of src, with optional padding
geom.rect_in_rect = (src, dst, padding = 0., axis = -1) => {
  var dst = geom.pad_rect(dst, padding);
  var [dst_w, dst_h] = mth.sub(dst[1], dst[0]);
  var [src_w, src_h] = mth.sub(src[1], src[0]);

  var ratiow = dst_w / src_w;
  var ratioh = dst_h / src_h;

  if (axis < 0) {
    if (ratiow <= ratioh)
      axis = 1;
    else
      axis = 0;
  }
  if (axis == 1) {  // fit vertically [==]
    var w = dst_w;
    var h = src_h * ratiow;
    var x = dst[0][0];
    var y = dst[0][1] + dst_h * 0.5 - h * 0.5;
  } else {        // fit horizontally [ || ]
    var w = src_w * ratioh;
    var h = dst_h;
    var y = dst[0][1];
    var x = dst[0][0] + dst_w * 0.5 - w * 0.5;
  }
  return geom.make_rect(x, y, w, h);
}

/// Return homogeneous transformation matrix that fits src rect into dst
geom.rect_in_rect_transform = (src, dst, padding = 0., axis = -1, scale=[1,1]) => {
  var fitted = geom.rect_in_rect(src, dst, padding, axis);
  var cenp_src = geom.rect_center(src);
  var cenp_dst = geom.rect_center(fitted);

  var M = mth.eye(3);
  M = mth.dot(M,
    geom.trans_2d(mth.sub(cenp_dst, cenp_src)));
  M = mth.dot(M, geom.trans_2d(cenp_src));
  M = mth.dot(M, geom.scaling_2d(scale));
  M = mth.dot(M, geom.scaling_2d(mth.div(geom.rect_size(fitted), geom.rect_size(src))));
  M = mth.dot(M, geom.trans_2d(mth.neg(cenp_src)));
  return M;
}

geom.rect_to_rect_transform = (src, dst) => {
  var [sw, sh] = geom.rect_size(src)
  var [dw, dh] = geom.rect_size(dst)

  var m = geom.trans_2d([dst[0][0], dst[0][1]])
  m = mth.dot(m, geom.scaling_2d([dw / sw, dh / sh]))
  m = mth.dot(m, geom.trans_2d([-src[0][0], -src[0][1]]))

  return m
}


/* measures */

geom.polygon_area = (P) => {
  if (P.length < 3)
    return 0;
  let area = 0.0;
  const n = P.length;
  let p0, p1;
  for (let i=0; i < n; i++){
    p0 = i;
    p1 = (i+1)%n;
    area += P[p0][0] * P[p1][1] - P[p1][0] * P[p0][1];
  }
  return area * 0.5;
}

/* Intersections and all */

geom.line_intersection = (s0, s1, eps = 1e-10) => {
  let sp0 = mth.cross([...s0[0], ...[1.0]], [...s0[1], ...[1.0]]);
  let sp1 = mth.cross([...s1[0], ...[1.0]], [...s1[1], ...[1.0]]);
  let ins = mth.cross(sp0, sp1);
  if (Math.abs(ins[2]) < eps)
    return [false, [0, 0]];
  return [true, [ins[0] / ins[2], ins[1] / ins[2]]];
}


geom.line_intersection_uv = (a1, a2, b1, b2, aIsSegment = false, bIsSegment = false) => {
  const EPS = 0.00001;
  var intersection = mth.zeros(2);
  var uv = mth.zeros(2)

  const denom = (b2[1] - b1[1]) * (a2[0] - a1[0]) - (b2[0] - b1[0]) * (a2[1] - a1[1]);
  const numera = (b2[0] - b1[0]) * (a1[1] - b1[1]) - (b2[1] - b1[1]) * (a1[0] - b1[0]);
  const numerb = (a2[0] - a1[0]) * (a1[1] - b1[1]) - (a2[1] - a1[1]) * (a1[0] - b1[0]);

  if (Math.abs(denom) < EPS)
    return [false, intersection, uv];

  uv[0] = numera / denom;
  uv[1] = numerb / denom;

  intersection[0] = a1[0] + uv[0] * (a2[0] - a1[0]);
  intersection[1] = a1[1] + uv[0] * (a2[1] - a1[1]);

  var isa = true;
  if (aIsSegment && (uv[0] < 0 || uv[0] > 1))
    isa = false;
  var isb = true;
  if (bIsSegment && (uv[1] < 0 || uv[1] > 1))
    isb = false;

  return [isa && isb, intersection, uv];
}

geom.line_segment_intersection = (a1, a2, b1, b2) => {
  var [res, intersection, uv] = geom.line_intersection_uv(a1, a2, b1, b2, false, true);
  return [res, intersection];
}

geom.segment_line_intersection = (a1, a2, b1, b2) => {
  var [res, intersection, uv] = geom.line_intersection_uv(a1, a2, b1, b2, true, false);
  return [res, intersection];
}

geom.segment_intersection = (a1, a2, b1, b2) => {
  var [res, intersection, uv] = geom.line_intersection_uv(a1, a2, b1, b2, true, true);
  return [res, intersection];
}

geom.line_ray_intersection = (a1, a2, b1, b2) => {
  var [res, intersection, uv] = geom.line_intersection_uv(a1, a2, b1, b2, false, false);
  return [res && uv[1] > 0, intersection];
}

geom.ray_line_intersection = (a1, a2, b1, b2) => {
  var [res, intersection, uv] = geom.line_intersection_uv(a1, a2, b1, b2, false, false);
  return [res && uv[0] > 0, intersection];
}

geom.ray_intersection = (a1, a2, b1, b2) => {
  var [res, intersection, uv] = geom.line_intersection_uv(a1, a2, b1, b2, false, false);
  return [res && uv[0] > 0 && uv[1] > 0, intersection];
}

geom.ray_segment_intersection = (a1, a2, b1, b2) => {
  var [res, intersection, uv] = geom.line_intersection_uv(a1, a2, b1, b2, false, true);
  return [res && uv[0] > 0 && uv[1] > 0, intersection];
}


geom.shapes = function() {}
geom.shapes.random_radial_polygon = (n, min_r, max_r, center = [0, 0]) => {
  let R = mth.uniform(min_r, max_r, n);
  let start = mth.uniform(0, Math.PI * 2);
  let Theta = mth.randspace(start, start + Math.PI * 2, n + 1).slice(0, n);
  var res = mth.transpose([mth.add(mth.mul(mth.cos(Theta), R), center[0]),
  mth.add(mth.mul(mth.sin(Theta), R), center[1])]);
  return res;
}

geom.shapes.star = (cenp, r, n=5, ratio_inner=1, start_theta=0) => {
  const R  = [r / (1.618033988749895 + 1) * ratio_inner, r];
  const theta = mth.add(mth.linspace(start_theta, Math.PI * 2 + start_theta, n * 2 + 1), Math.PI / (n * 2)).slice(0, n*2);
  var P = [];
  for (var i = 0; i < theta.length; i++)
    P.push([Math.cos(theta[i]) * R[i % 2] + cenp[0], Math.sin(theta[i]) * R[i % 2] + cenp[1]]);
  return P;
}


// Internal dp simplify
const dp_simplify = (X, eps, distfn = geom.point_line_distance) => {
  var stack = [];
  const n = X.length;
  stack.push([0, n - 1])
  var flags = _.range(0, n).map(v => true)

  while (stack.length) {
    var [a, b] = stack.pop();
    var dmax = 0.0;
    var index = a;

    for (var i = a + 1; i < b; i++) {
      if (flags[i]) {
        var d = distfn(X[i], X[a], X[b])
        if (d > dmax) {
          index = i;
          dmax = d
        }
      }
    }

    if (dmax > eps) {
      stack.push([a, index])
      stack.push([index, b])
    } else {
      for (var i = a + 1; i < b; i++)
        flags[i] = false
    }
  }
  return flags;
}

/**
 * Ramer-Douglas-Peucker polyline simplification (sorry Rammer)
 * @param {any} X
 * @param {any} eps
 * @param {bool} closed
 * @returns simplified polyline
 */
geom.dp_simplify = (X, eps, closed = false, get_inds = false, distfn = geom.point_line_distance) => {
  if (eps <= 0.) return ctr;

  if (closed)
    X = mth.vstack([X, [X[0]]]);

  var flags = dp_simplify(X, eps, distfn);
  var I = _.range(flags.length).filter(i => flags[i]);
  var Y = I.map(i => X[i]);
  if (closed) {
    Y = Y.slice(0, Y.length - 1);
    I = I.slice(0, I.length - 1);
  }

  if (get_inds)
    return [Y, I];
  return Y;
}


/**
 * Remove points along a contour that are closer together than eps
 * @param {Array} X contour
 * @param {bool} closed true if contour is closed
 * @param {number} eps min distance
 * @param {bool} get_inds get cleaned up contour indices
 * @returns cleaned up contour and (optionally) indices
 */

geom.cleanup_contour = (X, closed = false, eps = 1e-10, get_inds = false) => {
  const distfn = (p, a, b) => Math.max(mth.distance(p, a), mth.distance(p, b));
  return geom.dp_simplify(X, eps, closed, get_inds, distfn);
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

  if (C < 1){
    if (get_edge_inds)
      return [P_, _.range(0, P_.length)];
    else
      return P_;
  }

  let P = [...P_];
  if (closed){
    const a = P[P.length-1];
    const b = P[0];
    for (const t of mth.linspace(0, 1, 5).slice(1, 5))
      P.push(mth.lerp(a, b, t));

    //P.push(P[0]);
  }
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
      b0 = b1;
      b1 = b1.next;
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
  if (closed && Q.length > 2) {
    // Q = geom.cleanup_contour(Q, true, 0.01);
    // const [res, ins] = geom.line_intersection([Q[0], Q[1]], [Q[Q.length - 1], Q[Q.length - 2]])
    // if (res && mth.distance(ins, Q[0])<100) {
    //   Q[0] = ins;
    //   Q[Q.length - 1] = ins;
    // }
     //Q = Q.slice(0, Q.length-1)
     //edge_inds = edge_inds.slice(0, edge_inds.length-1)
  }
  if (get_edge_inds)
    return [Q, edge_inds];
  return Q;
}

const compare_nums = (a, b) => {
  return a - b;
}

/**
 * Computes hatch (scan)lines for a polygon or a compound shape according to the even-odd fill rule
 * @param {Array} polygon or shape
 * @param {Number} dist distance between scanlines
 * @param {Number} angle scanline orientation
 * @param {bool} flip_horizontal if true, flip odd scanline orientation
 * @param {any} max_count maximum number of scanlines
 * @param {any} eps tolerance for zero distance
 * @returns an array of coordinate pairs one pair for each scanline
 */
geom.hatch = (S, dist, angle=0.0, flip_horizontal=false, max_count=10000, eps=1e-10) => {
  if (!S.length)
    return [];
  if (!geom.is_compound(S))
    S = [S];


  // Rotate shape for oriented hatches
  const theta = mth.radians(angle);
  let mat = geom.rot_2d(-theta);
  S = geom.affine_transform(mat, S);

  const box = geom.bounding_box(S);

  // build edge table
  let ET = [];
  for (let i = 0; i < S.length; i++){
    const P = S[i];
    const n = P.length;
    if (n <= 2)
      continue;

    let dx, dy, m;
    for (let j = 0; j < n; j++){
      let [a, b] = [P[j], P[(j+1)%n]];
      // reorder increasing y
      if (a[1] > b[1])
        [a, b] = [b, a];
      // slope
      dx = (b[0] - a[0]);
      dy = (b[1] - a[1]);
      if (Math.abs(dx) > eps)
        m = dy/dx;
      else
        m = 1e15;
      if (Math.abs(m) < eps)
        m = null;
      ET.push({a:a, b:b, m:m, i:i});
    }
  }
  // sort by increasing y of first point
  ET.sort((ea, eb)=>compare_nums(ea.a[1], eb.a[1]));

  // intersection x
  const ex = (e, y) => {
    if (e.m == null)
      return null;
    return (e.a[0] + (y - e.a[1])/e.m);
  };


  let y = box[0][1];
  let scanlines = [];
  let AET = []; // active edge table
  let flip = 0;
  let c = 0;
  while (ET.length || AET.length){
    if (y > box[1][1])
      break;
    if (c >= max_count){
      console.log("scanlines: reached max number of iterations: " + c);
      break;
    }
    c += 1;

    // move from ET to AET
    let i = 0;
    for (const e of ET){
      if (e.a[1] <= y){
        AET.push(e);
        i += 1;
      }else{
        break;
      }
    }

    if (i < ET.length)
      ET = ET.slice(i);
    else
      ET = [];

    // remove passed edges
    AET.sort((ea, eb)=>compare_nums(ea.b[1], eb.b[1])); //  = sorted(AET, key=lambda e: e.b[1])
    AET = AET.filter(e=>e.b[1] > y);
    let xs = AET.map(e=>[ex(e, y), e.i]);
    xs = xs.filter(xi=>xi[0] != null);

    // sort xs (flipped each scanline for more efficent plotting )
    if (flip)
      xs.sort((va, vb)=>compare_nums(-va[0], -vb[0])); // = sorted(xs, key=lambda v: -v[0])
    else
      xs.sort((va, vb)=>compare_nums(va[0], vb[0]));

    if (flip_horizontal)
      flip = !flip;

    if (xs.length > 1){
      let parity = 1; // assume even-odd fill rule
      let x1,i1,x2,i2;
      for (let k = 0; k < xs.length-1; k++){
        [x1, i1] = xs[k];
        [x2, i2] = xs[k+1];
        let pa = [x1, y];
        let pb = [x2, y];

        if (parity){
          scanlines = scanlines.concat([pa,pb]);
        }

        parity = !parity;
      }
    }

    // increment
    y += dist;
  }

  // unrotate
  if (scanlines.length){
    scanlines = geom.affine_transform(mth.transpose(mat), scanlines);
    scanlines = _.range(0, scanlines.length, 2).map(i=>[scanlines[i], scanlines[i+1]]);
  }

  return scanlines;
}

geom.foo = () => 0;

module.exports = geom;
