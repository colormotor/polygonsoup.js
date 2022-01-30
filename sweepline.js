
/**
 *  ______ _______ _____  ___ ___ _______ _______ _______ _______ _______ _______ ______          _____ _______
 * |   __ \       |     ||   |   |     __|       |    |  |     __|       |   |   |   __ \______ _|     |     __|
 * |    __/   -   |       \     /|    |  |   -   |       |__     |   -   |   |   |    __/______|       |__     |
 * |___|  |_______|_______||___| |_______|_______|__|____|_______|_______|_______|___|         |_______|_______|
 *
 * © Daniel Berio (@colormotor) 2021 - ...
 *
 * Implementation of sweepline segment interesections.
 * Mainly based on Chapter 2 of De Berg et al. - Computational Geometry: Algorithms and Applications
 * with some ideas from:
 * Melhnorn and Naher (1994) Implementation of a Sweep Line Algorithm for the Straight Line Segment Intersection Problem
 * https://pure.mpg.de/rest/items/item_1834220/component/file_2035159/content
 * as well as (radial sorting for order) from an implementation here:
 * https://github.com/h2ssh/Vulcan
 **/

'use strict';
//import BinaryTree from 'avl';
const BinaryTree = require('avl').default;

const sweepline = function() { }

/* Fuzzy eqs */
const fequal = (a, b, eps = 1e-5) => Math.abs(a - b) <= eps;
const fleq = (a, b) => fequal(a, b) || a < b;
const fgeq = (a, b) => fequal(a, b) || a > b;

const to_point = (p) => [p[0], p[1]]
const make_isegment = (s) => [[...s[0], 1], [...s[1], 1]]

const line_intersection = (s0, s1) => {
  let sp0 = cross(s0[0], s0[1]);
  let sp1 = cross(s1[0], s1[1]);
  let ins = cross(sp0, sp1);
  if (fequal(ins[2], 0))
    return [false, [0, 0]];
  return [true, [ins[0] / ins[2], ins[1] / ins[2]]];
}

const point_in_box = (p, a, b) => {
  // if (p[0] <= Math.max(a[0], b[0]) && p[0] >= Math.min(a[0], b[0]) &&
  //   p[1] <= Math.max(a[1], b[1]) && p[1] >= Math.min(a[1], b[1]))
  //   return true;
  if (fleq(p[0], Math.max(a[0], b[0])) && fgeq(p[0], Math.min(a[0], b[0])) &&
      fleq(p[1], Math.max(a[1], b[1])) && fgeq(p[1], Math.min(a[1], b[1])))
    return true;
  return false;
}

const segment_intersection = (s0, s1) => {
  const [res, p] = line_intersection(s0, s1);

  const no = [false, [0, 0]]
  if (!res) {
    return no;
  }

  if (!point_in_box(p, s0[0], s0[1])) return no;
  if (!point_in_box(p, s1[0], s1[1])) return no;
  return [true, p];
}

/* vec utils */
const isub = (a, b) => [a[0] - b[0], a[1] - b[1], 1];
const iadd = (a, b) => [a[0] + b[0], a[1] + b[1], 1];
const idot = (a, b) => a[0] * b[0] + a[1] * b[1];
const iwedge = (v, w) => v[0] * w[1] - v[1] * w[0];
const cross = (a, b) => {
  let x = a[1] * b[2] - a[2] * b[1];
  let y = a[2] * b[0] - a[0] * b[2];
  let z = a[0] * b[1] - a[1] * b[0];
  return [x, y, z];
}

/* point and segment predicates */
const is_trivial = (s) => identical(s[0], s[1]);
const horizontal = (s) => fequal(s[0][1], s[1][1]);
const vertical = (s) => fequal(s[0][0], s[1][0]);
const identical = (a, b) => fequal(a[0], b[0]) && fequal(a[1], b[1]);
const segments_identical = (a, b) => {
  return (identical(a[0], b[0]) && identical(a[1], b[1])) ||
    (identical(a[0], b[1]) && identical(a[1], b[0]));
}


// comps (segment comparison is inside sweepline_intersections function)
const compare = (a, b) => {
  if (fequal(a, b))
    return 0;
  if (a < b)
    return -1;
  return 1;
}

const point_compare = (a, b) => {
  if (fequal(a[1], b[1])) {
    return compare(a[0], b[0]);
  }
  return compare(a[1], b[1]);
}

const x_intersection = (p_sweep, s) => {
  const dx = s[1][0] - s[0][0];
  const dy = s[1][1] - s[0][1];
  if (fequal(dy, 0)) return s[0][0];
  return (dx / dy) * (p_sweep[1] - s[0][1]) + s[0][0];
  return s
}


const segment_x = (s, p_sweep) => {
  let x;
  if (is_trivial(s)) {
    x = s[0][0];
  } else if (horizontal(s)) {
    x = s[0][0];
    if (x < p_sweep[0]) x = p_sweep[0];
  } else if (vertical(s)) {
    x = s[0][0];
  } else {
    if (identical(s[0], p_sweep))
      x = s[0][0];
    else if (identical(s[1], p_sweep))
      x = s[1][0];
    else {
      x = x_intersection(p_sweep, s);
    }
    s = s[0][0]
  }

  return x;
}

/* will need these for tree and sets */
const union = (setA, setB) => {
  let _union = new Set(setA)
  for (let elem of setB) {
    _union.add(elem)
  }
  return _union
}

const lower_bound = (Y, key) => {
  if (Y._root == null)
    return null;
  let cur = Y.minNode();
  let comparator = Y._comparator;
  while (cur != null) {
    if (comparator(cur.key, key) >= 0)
      return cur;
    cur = Y.next(cur);
  }
  return null;
}

const upper_bound = (Y, key) => {
  if (Y._root == null)
    return null;
  let cur = Y.minNode();
  let comparator = Y._comparator;
  while (cur != null) {
    if (comparator(cur.key, key) > 0)
      return cur;
    cur = Y.next(cur);
  }
  return null;
}

const equal_range = (Y, key) => {
  let range = [];
  let lo = lower_bound(Y, key);
  if (lo == null)
    return [];
  let cur = lo;
  while (cur != null && Y._comparator(cur.key, key) == 0) {
    range.push(cur);
    cur = Y.next(cur);
  }
  return range;
}


sweepline.edge_key = (a, b) => {
    if(a > b)
      return b + '_' + a;
    else
      return a + '_' + b;
  }

/**
 * Computes intersections between a list of segments each defined as [[x1,y1],[x2,y2]], using a variant of the Bentley Ottman method
 * @param {Array} segs list of segments
 * @param {bool} avoid_incident_endpoints (default true) don't consider incident segment points as an intersection
 * @param {int} debug_ind if > -1 populate debug info
 * @returns an object with the following entries:
 * - segments {Array} re-ordered segments for sweepline
 * - intersections {array}: with elements {pos: [x,y] intersection position, segments: segment indices}
 * - graph {object}: resulting planar graph represented as a series of vertices (positions) and edges (indexing positions)
 * - debug: additional debug info
 */
sweepline.sweepline_intersections = (segs, avoid_incident_endpoints = true, debug_ind = -1) => {
  let p_sweep = []; // Swept point
  let segments = []; // Input segments with ordered points

  let res = {
    debug : {
      ULCs: [],
      leftrights: [],
      step_keys: [],
      sweep_history: [],
      segments: [],
      intersections: [],
      stuck: false
    },
    segments:[], // reoriented segments
    intersections:[], // intersections
    graph: { // planar graph
      vertices: [],
      edges: [],
      edge_segments: []
    }
  }

  let brute_number = 0;
  let intersection_count = 0;

  let null_pt = [-100, -100, 1];
  let debug_pairs = [];

  // Planar graph  (internal)
  let node_map = {};
  let vertices = {};
  let edges = {};
  let edge_to_segment = {};

  const add_vertex = (v, pos) => {
    vertices[v] = pos;
  }

  const add_edge = (a, b, seg) => {
    var edge_id = sweepline.edge_key(a, b);
    if (!(edge_id in edges)){
      edges[edge_id] = [a, b];
      edge_to_segment[edge_id] = seg;
    }
  }

  let original = {};
  let flipped = {};
  let theta_segments = [];
  let guard_ids = [0, 0];

  var event_id = 0;

  const segment_compare = (i1, i2) => {
    const s1 = segments[i1];
    const s2 = segments[i2];

    let x1 = segment_x(s1, p_sweep),
        x2 = segment_x(s2, p_sweep);

    if (fequal(x1, x2))
      return 0;
    else if (x1 < x2)
      return -1;
    return 1;
  }

  const step = () => {
    if (X.isEmpty()) return false;
    let it = X.pop();
    handle_event_point(it);
    return true;
  }

  const handle_event_point = (it) => {
    let p = it.key;

    // insert U, C into Y. Ordering with respect to sweep is updated
    update_sweepline(p);
    // segments for current event (top point matches sweepline)
    let U = new Set();
    for (var i of it.data.S) {
        U.add(i);
    }

    // bottom-incident (L) and containing (C) segments
    let L = it.data.E;
    let C = new Set();
    find_swept_segments(L, C, U, p);

    // Update graph
    let vert = it.data.id;
    // Only if we are not dealing with a "guard" vertex
    if (!(vert >= guard_ids[0] && vert < guard_ids[1])){
      if (!p.length){
        console.log("problem");
      }
      add_vertex(vert, to_point(p));
      for (let i of C) {
        add_edge(node_map[i], vert, original[i]);
        node_map[i] = vert;
      }
      for (let i of L) {
        if (vert == undefined) {
          console.log('vert undefined');
          throw TypeError;
        }
        if (node_map[i] != undefined) {
          add_edge(node_map[i], vert, original[i]);

        } else {
          console.log('node_map undefined for ' + i);
          res.debug.stuck = true;
          //throw TypeError;
        }
        node_map[i] = vert;
      }
      for (let i of U)
        node_map[i] = vert;
    }

    // debug info
    if (debug_ind > -1)
      res.debug.ULCs.push({ U: U, L: L, C: C });

    if ((U.size + L.size + C.size) > 1) {
      // found intersection at p
      if (C.size || !avoid_incident_endpoints) {
        let ins = {
          pos: to_point(p),
          segments: new Set()
        };
        for (var i of U)
          if (point_in_box(p, segments[i][0], segments[i][1]))
            ins.segments.add(i);
        for (var i of L)
          if (point_in_box(p, segments[i][0], segments[i][1]))
            ins.segments.add(i);
        for (var i of C)
          if (point_in_box(p, segments[i][0], segments[i][1]))
            ins.segments.add(i);

        res.intersections.push(ins);
      }
    }

    // debug intersections up to this step
    if (debug_ind > -1)
      res.debug.intersections.push(res.intersections.slice(0));

    // Delete LC from Y
    for (var i of L) {
      Y.remove(i);
    }
    for (var i of C) {
      Y.remove(i);
    }

    // From https://github.com/h2ssh/Vulcan , DeBerg suggest point
    // "immidiately below" but this is equivalent to radially sorting the
    // segment endpoints around event point
    const UC = union(U, C);
    let theta_segments = radially_sorted_segments(p, UC);
    for (let v of theta_segments) {
      Y.insert(v.i);
    }

    // Debug current segments in status
    if (debug_ind > -1)
      res.debug.step_keys.push(Y.keys().filter(i=>i<segs.length));

    if (Y.isEmpty()) {
      return;
    }

    // for debug
    let lr = [null, null, []];

    if (UC.size) {
      let left = lower_bound(Y, query_sweepline());
      let leftleft = Y.prev(left);
      if (left && leftleft){
        let found = find_new_event(leftleft.key, left.key, p);
        if (leftleft.key < segs.length && left.key < segs.length) // check guards
          lr[0] = [leftleft.key, left.key];
      }
      let rightright = upper_bound(Y, query_sweepline());
      let right = Y.prev(rightright);
      if (right && rightright){
        let found = find_new_event(right.key, rightright.key, p);
        if (right.key < segs.length && rightright.key < segs.length) // check guards
          lr[1] = [right.key, rightright.key];
      }
      //if (debug_ind == res.leftrights.length)
      //  mth.print(lr);
    } else {
      // left and right neighbors of p
      let left = lower_bound(Y, query_sweepline());
      left = Y.prev(left);
      let right = upper_bound(Y, query_sweepline());
      if (left && right)
        {
          if (left.key < segs.length) // check guards
            lr[0] = [left.key];
          if (right.key < segs.length)
            lr[1] = [right.key];
          let r = find_new_event(left.key, right.key, p);
        }
    }

    if (debug_ind > -1)
      res.debug.leftrights.push(lr);
  }

  const find_new_event = (sl, sr, p) => {
    const [has_intersection, ins] = segment_intersection(segments[sl], segments[sr])
    if (has_intersection) {
      // sl sr intersect below the sweep line, or on it and to the right of the
      // current event point p, and the intersection is not yet present as an
      // event
      if (X.find(ins)) {
      // again must always use fuzzy equalities or things will go wrong with nearly horizontal segments
      //} else if (ins[1] > p[1] || (fequal(ins[1], p[1]) && ins[0] > p[0])) {
      } else if ((!fequal(ins[1], p[1]) && ins[1] > p[1]) ||
                  (fequal(ins[1], p[1]) && ins[0] > p[0])) {
        X.insert(ins, {S:new Set(), E:new Set(), id:event_id++});
        return true;
      }
    }

    return false;
  }

  const find_swept_segments = (L, C, U, p) => {
    const range = equal_range(Y, query_point(p));

    for (var node of range) {
      const seg = segments[node.key];
      if (!U.has(node.key)) {
        if (identical(seg[1], p_sweep)) {
          L.add(node.key); // Lower segment point on sweepline
        } else {
          C.add(node.key); // Sweepline on segment
        }
      }
        //else {
        // // U has the key but the segment may be horizontal
        // if (horizontal(seg) && identical(seg[1], p_sweep)){
        //   L.add(node.key);
        //   console.log('horizontal L');
        // }
        // }
    }
  }

  const update_sweepline = (p) => {
    p_sweep = p;
    if (debug_ind > -1)
      res.debug.sweep_history.push(p);
  }

  // update and get index for query point
  const query_point = (p) => {
    const i = segments.length - 1;
    segments[i] = [p, p];
    return i;
  }

  // sweepline index
  const query_sweepline = () => { return query_point(p_sweep); } //segments.length - 1; }

  const radially_sorted_segments = (origin, segs) => {
    let theta_segments = [];
    for (var i of segs) {
      const seg = segments[i];
      const p = seg[1];
      let theta = Math.atan2(p[1] - origin[1], p[0] - origin[0]) + Math.PI;
      theta_segments.push({ i: i, theta: theta });
    }

    return theta_segments.sort((a, b) => {
      if (a.theta < b.theta)
        return -1;
      if (a.theta > b.theta)
        return 1;
      return 0;
    });

    return theta_segments;
  }

  const add_segment_to_queue = (s, i) => {
    // add segment endpoints to queue
    var ptr = X.find(s[0]);
    if (ptr) {
      ptr.data.S.add(i); // Event present just add segments
    } else {
      X.insert(s[0], {S:new Set([i]), E:new Set(), id:event_id++});
    }

    ptr = X.find(s[1]);
    if (ptr) {
      ptr.data.E.add(i); // Event present just add segments
    } else {
      X.insert(s[1], {S:new Set(), E:new Set([i]), id:event_id++});
    }

    //X.insert(s[1], {S:new Set(), E:new Set(), id:event_id++});
  }


  /* procedure starts here */
  let X = new BinaryTree(point_compare, false);  // event queue
  let Y = new BinaryTree(segment_compare); // status

  // Add segments to queue
  let i = 0;
  let bounds = null;
  for (var k = 0; k < segs.length; k++) {
    let s = make_isegment(segs[k]);
    if (is_trivial(s)) continue;
    // Reorder segment vertically
    let flip = false;
    if ((!fequal(s[0][1], s[1][1]) && s[0][1] > s[1][1]) ||
         (fequal(s[0][1], s[1][1]) && s[0][0] > s[1][0])) {
      s = [s[1], s[0]];
      flip = true;
    }
    // get bounding box
    if (bounds == null){
      bounds = [[Math.min(s[0][0], s[1][0]), Math.min(s[0][1], s[1][1])],
                [Math.max(s[0][0], s[1][0]), Math.max(s[0][1], s[1][1])]];
    }else{
      bounds[0][0] = Math.min(bounds[0][0], Math.min(s[0][0], s[1][0]));
      bounds[0][1] = Math.min(bounds[0][1], Math.min(s[0][1], s[1][1]));
      bounds[1][0] = Math.max(bounds[1][0], Math.max(s[0][0], s[1][0]));
      bounds[1][1] = Math.max(bounds[1][1], Math.max(s[0][1], s[1][1]));
    }
    // input_segments.push_back(segs[i]);
    segments.push(s);
    flipped[k] = flip;
    original[i] = k;

    add_segment_to_queue(s, i);
    i++;
  }

  // store re-oriented segments
  res.segments = segments;

  // add "guard" vertical segments that will never intersect
  // (this helps with corner cases involving horizontal segments)
  guard_ids[0] = event_id; // store id so we don't create graph vertices for these

  const offset = 10000
  segments.push([[bounds[0][0]-offset, bounds[0][1]-offset, 1],
                 [bounds[0][0]-offset, bounds[1][1]+offset, 1]])
  add_segment_to_queue(segments[segments.length-1], i++);

  segments.push([[bounds[1][0]+offset, bounds[0][1]-offset, 1],
                 [bounds[1][0]+offset, bounds[1][1]+offset, 1]])
  add_segment_to_queue(segments[segments.length-1], i++);
  guard_ids[1] = event_id;
  // last segment always contains a query point, used to test points during sweep
  segments.push([p_sweep, p_sweep]);

  // iterate through event-queue
  let nsteps = 0;
  while (step()) {
    nsteps++;
    // uncomment to debug when intersection testing gets stuck in an infinite loop
    // guard segments should avoid that
    // if (debug_ind > -1 && nsteps > 3000){
    //   res.debug.stuck = true;
    //   break;
    // }
  }

  // construct output planar graph
  let nverts = 0;
  let vertex_index = {};
  let count = 0;
  for (let v in vertices){
    res.graph.vertices.push(vertices[v]);
    vertex_index[v] = count++;
  }

  // for (let v in vertices){
  //   res.graph.vertices[v] = vertices[v];
  // }
  // console.log(res.graph.vertices.length);
  // for (const v of res.graph.vertices)
  //   if (!v.length)
  //     console.log(v.length);

  for (let id in edges){
    let [a,b] = edges[id];

    res.graph.edges.push([vertex_index[a], vertex_index[b]]); //edges[id]);
    res.graph.edge_segments.push(edge_to_segment[id]);
  }

  return res;
}


module.exports = sweepline;
//export default sweepline;
