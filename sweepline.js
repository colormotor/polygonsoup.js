'use strict';
const mth = require('./mth.js')
//import mth from 'mth';
import AVLTree from 'avl';
//const AVL = require("avl");

//console.log(new AVL());

const sweepline = function() { }

const fequal = (a, b, eps = 1e-4) => {
  return Math.abs(a - b) <= eps;
}

const _x = (v) => v[0]
const _y = (v) => v[1]
const _x1 = (s) => _x(s[0])
const _x2 = (s) => _x(s[1])
const _y1 = (s) => _y(s[0])
const _y2 = (s) => _y(s[1])
const _dX = (s) => (_x(s[1]) - _x(s[0]))
const _dY = (s) => (_y(s[1]) - _y(s[0]))

// Modify the first two if the imput type uses for example [0], [1] for coords

const to_point = (p) => [p[0], p[1]]

const make_isegment = (s) => [[...s[0], 1], [...s[1], 1]]

const line_intersection = (s0, s1) => {
  let sp0 = cross(s0[0], s0[1]);
  let sp1 = cross(s1[0], s1[1]);
  let ins = cross(sp0, sp1)
  return [true, [ins[0] / ins[2], ins[1] / ins[2]]];
}

const point_in_box = (p, a, b) => {
  // TODO check non-rational precision here
  if (p[0] <= Math.max(a[0], b[0]) && p[0] >= Math.min(a[0], b[0]) &&
    p[1] <= Math.max(a[1], b[1]) && p[1] >= Math.min(a[1], b[1]))
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

const horizontal = (s) => fequal(s[0][1], s[1][1]);

const identical = (a, b) => fequal(a[0], b[0]) && fequal(a[1], b[1]);

const segments_identical = (a, b) => {
  return (identical(a[0], b[0]) && identical(a[1], b[1])) ||
    (identical(a[0], b[1]) && identical(a[1], b[0]));
}

const is_trivial = (s) => identical(s[0], s[1]);

// comps
const compare = (a, b) => {
  if (fequal(a, b))
    return 0;
  if (a < b)
    return -1;
  return 1;
}

const point_less = (a, b) => {
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

// Sweepline ordering

const sweep_event = function(S, id) {
  this.S = S;
  this.id = id;
}

sweepline.sweepline_intersections = (segs, debug_ind = -1, avoid_incident_endpoints = true, p5 = null) => {
  let intersections = [];

  let p_sweep = [];

  //std::map<int, int>     original;
  let segments = [];
  let flipped = [];
  // let sweep_history = [];
  // let ULCs = []
  // let leftrights = []
  // let step_keys = []

  let res = {
    ULCs: [],
    leftrights: [],
    step_keys: [],
    sweep_history: [],
    segments: [],
    intersections: []
  }

  let brute_number = 0;
  let intersection_count = 0;

  let null_pt = [-100, -100, 1];
  let debug_pairs = [];
  let debug_new_ins = null_pt;

  //SweepGraph*        graph;
  //std::map<int, int> node_map;

  let theta_segments = [];

  let event_id = 0;

  const segment_less = (i1, i2) => {
    const s1 = segments[i1];
    const s2 = segments[i2];

    // last segment in list holds sweepline
    //const sweepline =
    const psweep = p_sweep; //segments[get_sweepline()][0];// p_sweep; //segments[segments.length - 1][0];

    const sweep_x = psweep[0];

    //console.log('less ' + i1);// p_segments[p_segments.length-1]);
    let x1 = 0, x2 = 0;
    if (is_trivial(s1)) {
      x1 = s1[0][0];
    } else if (horizontal(s1)) {
      x1 = s1[0][0];
      if (x1 < sweep_x) x1 = sweep_x;
    } else {
      if (identical(s1[0], psweep))
        x1 = s1[0][0];
      else if (identical(s1[1], psweep))
        x1 = s1[1][0];
      else {
        x1 = x_intersection(psweep, s1);

      }
    }

    if (is_trivial(s2)) {
      x2 = s2[0][0];
    } else if (horizontal(s2)) {
      x2 = s2[0][0];
      if (x2 < sweep_x) x2 = sweep_x;
    } else {
      if (identical(s2[0], psweep))
        x2 = s2[0][0];
      else if (identical(s2[1], psweep))
        x2 = s2[1][0];
      else {
        x2 = x_intersection(psweep, s2);
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

  let X = new AVLTree(point_less, true);  // event queue
  let Y = new AVLTree(segment_less, false); // status

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
    //if (graph) graph->flipped[k] = flipped;
    // original[i] = k;
    // add segment endpoints to queue
    var ptr = X.find(s[0]);
    if (ptr) {

      ptr.data.S.add(i);
    } else {
      X.insert(s[0], new sweep_event(new Set([i], event_id++)));
    }
    X.insert(s[1], new sweep_event(new Set(), event_id++));
    i++;
  }

  res.segments = segments;

  // second to last segment always contains a query point, used to test points during sweep
  segments.push([p_sweep, p_sweep]);
  // last segment always contains sweepline, so it can be accessed in
  // sweep_less and stored as an index
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
    // int vert = it->second.id;
    // if (graph) {
    //   graph->add_vertex(vert, to_point(p));

    //   for (auto i : C) {
    //     graph->add_edge(node_map[i], vert, original[i]);
    //     node_map[i] = vert;
    //   }

    //   for (auto i : L) {
    //     graph->add_edge(node_map[i], vert, original[i]);
    //     node_map[i] = vert;
    //   }

    //   for (auto i : U) node_map[i] = vert;
    // }

    // update last nodes

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
    for (var v of theta_segments) {
      Y.insert(v.i);
    }

    // Debug current segments in status
    res.step_keys.push(Y.keys());

    if (Y.isEmpty()) {
      return;
    }

    let lr = [null, null, []] //range.map(n => n.key)]
    // Steps 8-16
    if (UC.size) {
      let left = lower_bound(Y, get_sweepline());
      let leftleft = Y.prev(left);
      if (left && leftleft){
        find_new_event(leftleft.key, left.key, p);
        lr[0] = [leftleft.key, left.key];
      }
      let rightright = upper_bound(Y, get_sweepline());
      let right = Y.prev(rightright);
      if (right && rightright){
        find_new_event(right.key, rightright.key, p);
        lr[1] = [right.key, rightright.key];
      }
    } else {
      // left and right neighbors of p
      let left = lower_bound(Y, get_sweepline());
      left = Y.prev(left);
      let right = upper_bound(Y, get_sweepline());
      if (left && right)
        {
          lr[0] = left.key;
          lr[1] = right.key;
          let r = find_new_event(left.key, right.key, p);
        }
    }

    res.leftrights.push(lr);
  }



  // const equal_range = (Y, key) => {
  //   let n = Y.find(key);
  //   if (n == null){
  //     n = Y._root;
  //     while(n){
  //       if
  //     }

  //     return []
  //   }
  //   let range = [n];
  //   n = Y.prev(range[0]);
  //   while (n) {
  //     if (Y._comparator(n.key, key) != 0)
  //       break;
  //     range.push(n);
  //     n = Y.prev(n);
  //   }

  //   n = Y.next(range[0]);
  //   while (n) {
  //     if (Y._comparator(n.key, key) != 0)
  //       break;
  //     range.push(n);
  //     n = Y.next(n);
  //   }

  //   return range;
  // }


  const find_new_event = (sl, sr, p) => {
    //ipoint ins;
    //debug_pairs.push_back({segments[sl], segments[sr]});
    //debug_new_ins = {-100, -100, 1};
    if (sl == sr)
      console.log('nope');
    const [res, ins] = segment_intersection(segments[sl], segments[sr])

    if (res) {
      //debug_new_ins = ins;
      if (X.find(ins)) { //} != X.end()) {
        // std::cout << "ipoint already present...\n";
      } else if (ins[1] > p[1] || (fequal(ins[1], p[1]) && ins[0] > p[0])) {
        X.insert(ins, new sweep_event(new Set(), event_id++));
        return true;
      }
    }

    return false;
  }


  const find_swept_segments = (L, C, U, p) => {
    const range = equal_range(Y, query_point(p));
    for (var node of range) {
      if (!U.has(node.key)) {
        const seg = segments[node.key];
        if (identical(seg[1], p_sweep)) {
          L.add(node.key);
        } else {
          C.add(node.key);
        }
      }
    }
  }

  const update_sweepline = (p) => {
    p_sweep = p;
    segments[get_sweepline()] = [p_sweep, p_sweep];
    res.sweep_history.push(p);
    //console.log(segments[get_sweepline()]);
  }

  // update and get index for query point
  const query_point = (p) => {
    const i = segments.length - 2;
    segments[i] = [p, p];
    return i;
  }

  // sweepline index
  const get_sweepline = () => { segments.length - 1; }

  const radially_sorted_segments = (origin, segs) => {
    let theta_segments = [];
    for (var i of segs) {
      const seg = segments[i];
      const p = seg[1];
      const theta = Math.atan2(p[1] - origin[1], p[0] - origin[0]) + Math.PI;
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

  while (step()) {
  }

  // let test = new AVLTree(point_less);
  // test.insert([0, 0]);
  // test.insert([0, 0]);
  // test.insert([1, 1]);
  // test.insert([1, 1]);
  // test.insert([1, 1]);
  // test.insert([1, 2]);
  // test.insert([3, 2]);
  // let range = equal_range(test, [1, 1]);
  // mth.print(range.map(n => n.key));
  return [intersections, res]
}


// void planar_graph(SweepGraph*                   graph,
//                   const std::vector<segment_t>& segs,
//                   bool                          avoid_incident_endpoints) {
//   Sweepline sweep(segs, avoid_incident_endpoints, graph, false);
//   sweep.compute();
// }

// std::vector<Intersection> sweepline_intersections(
//     const std::vector<segment_t>& segs,
//     SweepGraph*                   graph,
//     bool                          avoid_incident_endpoints,
//     bool                          debug_brute_force) {
//   Sweepline sweep(segs, avoid_incident_endpoints, graph);
//   sweep.compute();
//   if (debug_brute_force) sweep.brute_force_debug();
//   return sweep.intersections;
// }

// std::vector<SegmentIntersection> sweepline_segment_intersections(
//     const std::vector<segment_t>& segs,
//     bool                          avoid_incident_endpoints,
//     bool                          debug_brute_force) {
//   std::vector<Intersection> intersections =
//       sweepline_intersections(segs,
//                               0,
//                               avoid_incident_endpoints,
//                               debug_brute_force);
//   std::vector<SegmentIntersection> segment_intersections;

//   std::set<std::pair<int, int>> visited;
//   for (int i = 0; i < intersections.size(); i++) {
//     const Intersection& ins = intersections[i];
//     for (auto it = ins.segments.begin(); it != ins.segments.end(); ++it) {
//       for (auto jt = std::next(it); jt != ins.segments.end(); ++jt) {
//         std::pair<int, int> ab = {*it, *jt};
//         if (visited.find(ab) != visited.end()) continue;
//         visited.add(ab);
//         SegmentIntersection sins;
//         sins.pos    = ins.pos;
//         sins.seg[0] = ab[0];
//         sins.seg[1] = ab[1];
//         segment_intersections.push(sins);
//       }
//     }
//   }

//   return segment_intersections;
// }

//module.exports = sweepline;
export default sweepline;
