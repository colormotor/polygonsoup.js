'use strict';
const mth = require('./mth.js')
//import mth from 'mth';
import BinaryTree from 'avl';
//import BinaryTree from 'splaytree';

const sweepline = function() { }

const fequal = (a, b, eps = 1e-10) => Math.abs(a - b) <= eps;
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

const point_on_segment = (r, s) => {
  let p = s[0];
  let q = s[1];
  let v = isub(q, p);
  let w = isub(r, p);
  return fequal(wedge(v, w), 0) && idot(v, w) >= 0;
}

const is_trivial = (s) => identical(s[0], s[1]);
const horizontal = (s) => fequal(s[0][1], s[1][1]);
const identical = (a, b) => fequal(a[0], b[0]) && fequal(a[1], b[1]);
const segments_identical = (a, b) => {
  return (identical(a[0], b[0]) && identical(a[1], b[1])) ||
    (identical(a[0], b[1]) && identical(a[1], b[0]));
}


// comps
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
}

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

const sweep_event = function(S, id) {
  this.S = S;
  this.id = id;
}

sweepline.sweepline_intersections = (segs, debug_ind = -1, avoid_incident_endpoints = true, graph = null) => {
  let intersections = [];
  let p_sweep = []; // Swept point
  let segments = []; // Input segments with ordered points

  let res = {
    ULCs: [],
    leftrights: [],
    step_keys: [],
    sweep_history: [],
    segments: [],
    intersections: [],
    stuck: false
  }

  let brute_number = 0;
  let intersection_count = 0;

  let null_pt = [-100, -100, 1];
  let debug_pairs = [];

  // Planar graph 
  let vertices = {};
  let edges = {};
  let node_map = {};

  const add_vertex = (v, pos) => {
    vertices[v] = pos;
  }

  const add_edge = (a, b) => {
    var edge_id = a + '_' + b;
    if(a > b) 
      edge_id = b + '_' + a;
    if (!(edge_id in edges)){
      edges[edge_id] = [a, b];
    }
  }
  
  let original = {};
  let flipped = {};
  let theta_segments = [];

  var event_id = 0;

  const segment_compare = (i1, i2) => {
    const s1 = segments[i1];
    const s2 = segments[i2];

    const sweep_x = p_sweep[0];

    let x1 = 0, x2 = 0;
    if (is_trivial(s1)) {
      x1 = s1[0][0];
    } else if (horizontal(s1)) {
      x1 = s1[0][0];
      //if (x1 < sweep_x) x1 = sweep_x; 
    } else {
      if (identical(s1[0], p_sweep))
        x1 = s1[0][0];
      else if (identical(s1[1], p_sweep))
        x1 = s1[1][0];
      else {
        x1 = x_intersection(p_sweep, s1);
      }
    }

    if (is_trivial(s2)) {
      x2 = s2[0][0];
    } else if (horizontal(s2)) {
      x2 = s2[0][0];
      //if (x2 < sweep_x) x2 = sweep_x;
    } else {
      if (identical(s2[0], p_sweep))
        x2 = s2[0][0];
      else if (identical(s2[1], p_sweep))
        x2 = s2[1][0];
      else {
        x2 = x_intersection(p_sweep, s2);
      }
    }

    let res = 0;

    if (fequal(x1, x2)) {
      res = 0;
    } else if (x1 < x2)
      res = -1;
    else
      res = 1;
    return res;
  }

  let X = new BinaryTree(point_compare, false);  // event queue
  let Y = new BinaryTree(segment_compare); // status

  // Add segments to queue
  let i = 0;
  for (var k = 0; k < segs.length; k++) {
    let s = make_isegment(segs[k]);
    if (is_trivial(s)) continue;
    // Reorder segment vertically
    let flip = false;
    if (s[0][1] > s[1][1] ||
      (fequal(s[0][1], s[1][1]) && s[0][0] > s[1][0])) {
      s = [s[1], s[0]];
      flip = true;
    }
    // input_segments.push_back(segs[i]);
    segments.push(s);
    flipped[k] = flip;
    original[i] = k;

    // add segment endpoints to queue
    var ptr = X.find(s[0]);
    if (ptr) {
      ptr.data.S.add(i); // Event present just add segments
    } else {
      X.insert(s[0], {S:new Set([i]), id:event_id++}); //new sweep_event(new Set([i], event_id++)));  
    }
    
    X.insert(s[1], {S:new Set(), id:event_id++}); //new sweep_event(new Set(), event_id++));
    
    i++;
  }

  res.segments = segments;

  // last segment always contains a query point, used to test points during sweep
  segments.push([p_sweep, p_sweep]);
  
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
      let seg = segments[i];
      if (identical(seg[0], p_sweep))
        U.add(i);
    }

    // bottom-incident (L) and containing (C) segments
    let L = new Set(), C = new Set();
    find_swept_segments(L, C, U, p);
    // // Update graph
    // // Add vertex to graph
    const vert = it.data.id;
    // console.log(node_map);
    add_vertex(vert, to_point(p));
      for (let i of C) {
        add_edge(node_map[i], vert);
        node_map[i] = vert;
      }
      for (let i of L) {
        add_edge(node_map[i], vert);
        node_map[i] = vert;
      }
      for (let i of U)
        node_map[i] = vert;
    
    // debug info
    res.ULCs.push({ U: U, L: L, C: C });

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

        intersections.push(ins);
      }
    }

    // debug intersections up to this step
    res.intersections.push(intersections.slice(0));

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
    res.step_keys.push(Y.keys());
    
    if (Y.isEmpty()) {
      return;
    }

    let lr = [null, null, []] ;
    
    if (UC.size) {
      let left = lower_bound(Y, query_sweepline());
      let leftleft = Y.prev(left);
      if (left && leftleft){
        let found = find_new_event(leftleft.key, left.key, p);
        lr[0] = [leftleft.key, left.key];
      }
      let rightright = upper_bound(Y, query_sweepline());
      let right = Y.prev(rightright);
      if (right && rightright){
        let found = find_new_event(right.key, rightright.key, p);
        lr[1] = [right.key, rightright.key];
      }
    } else {
      // left and right neighbors of p
      let left = lower_bound(Y, query_sweepline());
      left = Y.prev(left);
      let right = upper_bound(Y, query_sweepline());
      if (left && right)
        {
          lr[0] = [left.key];
          lr[1] = [right.key];
          let r = find_new_event(left.key, right.key, p);
        }
    }

    res.leftrights.push(lr);
  }

  const find_new_event = (sl, sr, p) => {
    const [has_intersection, ins] = segment_intersection(segments[sl], segments[sr])
    if (has_intersection) {
      // sl sr intersect below the sweep line, or on it and to the right of the
      // current event point p, and the intersection is not yet present as an
      // event
      if (X.find(ins)) { 
      } else if (ins[1] > p[1] || (fequal(ins[1], p[1]) && ins[0] > p[0])) {
        X.insert(ins, {S:new Set(), id:event_id++}); 
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
      } else {
        // U has the key but the segment may be horizontal
        if (horizontal(seg) && identical(seg[1], p_sweep)){
          L.add(node.key);
          console.log('horizontal L');
        }
      }
    }
  }

  const update_sweepline = (p) => {
    p_sweep = p;
    //segments[get_sweepline()] = [p_sweep, p_sweep];
    res.sweep_history.push(p);
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

  let nsteps = 0;
  while (step()) {
    nsteps++;
    // if (nsteps > 1300){
    //   res.stuck = true;
    //   break;
    // }
  }
  
  let nverts = 0;
  res.verts = [];
  for (let v in vertices){
    res.verts.push([]);
  }
  for (let v in vertices){
    res.verts[v] = vertices[v];
  } 
  
  res.edges = [];
  for (let id in edges){
    res.edges.push(edges[id]);
  }
  //
  // let tree = new BinaryTree();
  // tree.insert(0);
  // tree.insert(10);
  // tree.insert(10);
  // tree.insert(3);
  // tree.insert(20);
  // console.log(equal_range(tree, 10).map(n=>n.key)) //  upper_bound(tree, 10)); 
  return [intersections, res]
}

export default sweepline;
