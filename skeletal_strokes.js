"use strict"
const alg = require("./alg.js");
const geom = require("./geom.js");
const mth = require("./mth.js");

const skeletal_strokes = function() {}

/* Fuzzy eqs */
const zero_eps = 0;//1e-5;
const fequal = (a, b) => Math.abs(a - b) <= zero_eps;
const fleq = (a, b) => fequal(a, b) || a < b;
const fgeq = (a, b) => fequal(a, b) || a > b;
const fless = (a, b) => a < b-zero_eps;

const Edge = (i, a, b, m, flipped) => {
  return {i: i,
          a: a,
          b: b,
          m: m,
          flipped: flipped };
}

const FleshVertex = (pos,
                     type="NORMAL",
                     partition_index = -1,
                     spine_index = -1,
                     flipped = false) => {
                       return {pos: pos,
                               type: type,
                               partition_index: partition_index,
                               spine_index: spine_index,
                               flipped: flipped,
                               convave: false,
                               retrograe: false,
                               prev: null,
                               next: null};
                     };

const compare_nums = (a, b) => {
  return a - b;
};

/**
 * Splits a polyline at a given sequence of horizontal positions.
 * Outputs a linked list where vertices corresponding to intersections (JOINTS)
 * are duplicated This is critical when used in the skeletal stroke algorithm
 * to handle mitering at corners.
 * @param {any} P
 * @param {any} closed
 * @param {any} xs
 * @param {any} eps
 * @param {any} flip_normals
 * @param {any} duplicate_joints
 * @returns
 */
skeletal_strokes.split_polyline_horizontal = (p5, P,
  closed,
  xs,
  duplicate_joints) => {
  let edges = geom.edges(P, closed);
  let N = geom.normals_2d(P, closed, false);

  let ET = [];
  let a, b;
  for (let i = 0; i < edges.length; i++) {
    let e = edges[i];
    let flip = false;

    // reverse and flag edge if so it is always oriented with increasing x
    if (P[e[0]][0] > P[e[1]][0]) {
      a = e[1];
      b = e[0];
      flip = true;
    } else {
      a = e[0];
      b = e[1];
    }

    // slope
    let dx = P[b][0] - P[a][0];
    let dy = P[b][1] - P[a][1];
    let m;
    if (Math.abs(dx) > zero_eps)
      m = dy / dx;
    else
      m = Math.inf;
    ET.push(Edge(i, a, b, m, flip));
  }

  // sort by increasing x of first point
  // ET.sort((a, b) => {
  //   if (fequal(P[a.a][0], P[b.a][0]))
  //     return 0;
  //   if (P[a.a][0] < P[b.a][0])
  //     return -1;
  //   return 1;
  // });
  ET.sort((a,b)=>compare_nums(P[a.a][0], P[b.a][0]));

  let AET = [];  // active edge table

  let flesh_data = [];
  let vlist = new alg.DoublyLinkedList();
  for (let i = 0; i < P.length; i++) {
    flesh_data.push(FleshVertex(P[i]));

    if (i < N.length)
      flesh_data[i].edge_normal = N[i];
    else
      flesh_data[i].edge_normal = N[i - 1];

    vlist.add(flesh_data[i]);
  }

  // Scan horizontal subdivisions
  // First sort horizontally and remove boundary segments (it might be a good
  // idea to add this as a flag?)

    let xs_sorted = [...xs].sort((a,b)=>compare_nums(a,b));

    if (xs_sorted.length > 2)
      xs_sorted = xs_sorted.slice(1, xs.length - 1);
  // else
  //   xs_sorted = [];

  let m = xs_sorted.length;  // P.length;

  const CLAMP_PARTITION = (i) => parseInt(Math.max(Math.min(i, m), 0));

  for (let i = 0; i < m; i++) {
    let x = xs_sorted[i];
    // select move active edges from ET to AET
    let c = 0;
    for (const e of ET) {
      if (fleq(P[e.a][0], x)) {
        AET.push(e);
        c += 1;
      } else {
        break;
      }
    }

    if (c < ET.length)
      ET = ET.slice(c);
    // for (var j = 0; j < c; j++)
    //   if (ET.length) ET.shift();

    // sort AET according to second point
    // AET.sort((a, b) => {
    //   if (fequal(P[a.b][0], P[b.b][0]))
    //     return 0;
    //   if (P[a.b][0] < P[b.b][0])
    //     return -1;
    //   return 1;
    // });
    AET.sort((a,b)=>compare_nums(P[a.b][0], P[b.b][0]));
    //AET = AET.filter(e=>P[e.b][1] >= x);

    for (const e of AET) {
      let has_intersection = true;
      // first test if intersection is within a given epsilon range of a vertex,
      // if so we will just adjust the vertex position and not add a new
      // intersection

      if (fequal(P[e.a][0], x)) {
        flesh_data[e.a].type = "JOINT";
        flesh_data[e.a].pos[0] = x;
        if (flesh_data[e.a].partition_index == -1) {
          flesh_data[e.a].partition_index = CLAMP_PARTITION(i);
          flesh_data[e.a].flipped = e.flipped;
        }
        has_intersection = false;

      } else if (fequal(P[e.b][0], x)) {
        flesh_data[e.b].type = "JOINT";
        flesh_data[e.b].pos[0] = x;
        if (flesh_data[e.a].partition_index == -1) {
          flesh_data[e.a].partition_index = CLAMP_PARTITION(i);
          flesh_data[e.a].flipped = e.flipped;
        }
        has_intersection = false;
      }

      // we may have passed this segment all together,
      // so also in this case there will be no intersection
      if (fleq(P[e.b][0], x)) {
        if (flesh_data[e.b].partition_index == -1) {
          flesh_data[e.b].partition_index = CLAMP_PARTITION(i);
          flesh_data[e.b].flipped = e.flipped;
        }
        has_intersection = false;
      }

      // setup info for the first edge vertex the first time it is encountered
      if (fleq(P[e.a][0], x)) {
        // make sure partition are set
        if (flesh_data[e.a].partition_index == -1) {
          flesh_data[e.a].partition_index = CLAMP_PARTITION(i);
          flesh_data[e.a].flipped = e.flipped;
        }
      }

      // and finally sort out intersections if any
      if (has_intersection) {
        // insert y intersection
        let y = (e.m == Math.inf) ? P[e.a][1] : e.m * (x - P[e.a][0]) + P[e.a][1];
        let v = FleshVertex([x, y],
          "JOINT",
          CLAMP_PARTITION(i),
          i,
          e.flipped);
        v.edge_normal = N[e.i];

        // p5.fill(0,255,0);
        // console.log(x);
        // p5.circle(x, y , 20);
        if (e.flipped){
          vlist.add(v, flesh_data[e.b]);
        }else{
          vlist.insert(v, flesh_data[e.b]);
        }
      }
    }


    AET = AET.filter(e=>P[e.b][0] >= x);
    // // Remove passed edges from AEt
    // for (const e of AET) {
    //   if (fless(P[e.b][0], x)) {
    //     AET.shift();
    //   } else {
    //     break;
    //   }
    // }
  }

  // make sure we did not miss the final vertices
  let v = vlist.front;
  while (v) {
    if (v.partition_index == -1) v.partition_index = m;
    v = v.next;
  }

  // Now duplicate all joint vertices, and set spine indices
  v = vlist.front;
  if (duplicate_joints) {
    while (v) {
      let next = v.next;
      if (v.type == "JOINT") {
        v.spine_index = v.partition_index + 1;
        let vdup = _.clone(v);
        vlist.add(vdup, v);
        if (v.flipped) {
          v.partition_index = CLAMP_PARTITION(v.partition_index + 1);
        } else {
          vdup.partition_index = CLAMP_PARTITION(vdup.partition_index + 1);
        }
      }
      v = next;
    }
  }

  let verts = [];
  v = vlist.front;
  while (v) {
    if (mth.has_nan(v.pos))
      console.log("Pos has Nan");
    else
      verts.push(v);
    v = v.next;
  }

  return verts;
}

skeletal_strokes.stroke = (prototype, spine, width_profile, closed=false, rect=null, params={}) => {
  if (params.miter_limit==undefined) params.miter_limit = 1.5;
  if (params.unfold==undefined) params.unfold = false;
  if (params.fold_angle_sigma==undefined) params.fold_angle_sigma = 1.5;
  if (params.duplicate_joints==undefined) params.duplicate_joints = true;
  let debug = false;
  let p5 = null;
  if (params.debug!=undefined)
    p5 = params.debug;
  if (rect==null)
    rect = geom.bounding_box(prototype);

  const proto_box = geom.bounding_box(prototype);

  const aspect_l = fabs(geom.rect_l(rect) - geom.rect_l(proto_box)) / geom.rect_h(rect);
  const aspect_r = fabs(geom.rect_r(proto_box) - geom.rect_r(rect)) / geom.rect_h(rect);

  // normalize control path and widths
  const    ch_lengths      = geom.cum_chord_lengths(spine);
  const    segment_lengths = geom.chord_lengths(spine);
  const l               = geom.chord_length(spine);
  if (l < zero_eps)
    return;

  let P = spine;
  let W = width_profile;
  if (mth.dim(width_profile).length < 1)
    W = _.range(0, W.length-1).map(i=>[W[i], W[i+1]]);
  W = mth.mul(W, geom.rect_h(rect));

  const edges = geom.edges(P, closed); // TODO closed

  // normals, bisectors and other values
  let D = mth.diff(P, 0);
  let N = D.map(d=>mth.neg(mth.perp(mth.normalize(d))));

  let Alpha = geom.turning_angles(P, closed, true);  // P.closed, true);
  if (mth.has_nan(Alpha))
    console.log( "NaN turning angle" );

  let xsub = mth.div(ch_lengths, l);  // x subdibisions

  if (mth.has_nan(xsub))
    console.log("NaN xsub");

  // Compute frames for each joint of the spine
  const m = D.length;
  let frames = [];

  let frame_count = m;
  if (!closed)
    frame_count -= 1;
  for (let i = 0; i < frame_count; i++) {
    let p = P[(i + 1) % m];

    let    d1 = mth.normalize(D[i]);
    let    d2 = mth.normalize(D[(i + 1) % m]);
    let    w1 = W[i][1];
    let    w2 = W[(i+1)%m][0];

    let   alpha = Alpha[(i + 1) % m];
    // std::cout << "alpha: " << alpha << std::endl;

    if (Math.abs(alpha) < 1e-5) {
      alpha = 1e-5;
    }

    let o1 = w2 / Math.sin(alpha);  // eq 2
    let o2 = w1 / Math.sin(alpha);

    if (Math.has_nan(o1) || Math.has_nan(o2))
      console.log("sin(alpha)==0");

    let u1 = mth.mul(d1, o1 * sign(alpha));  // eq 1
    let u2 = mth.mul(d2, -o2 * sign(alpha));

    frames.push([u1, u2]);
  }

  // compute left and right profiles
  let L = [];
  let R = [];

  // This will get replaced if we have a closed polyline
  L.push(mth.add(P[0], mth.mul(N[0], W[0][0])));
  R.push(mth.sub(P[0], mth.mul(N[0], W[0][0])));

  for (let i = 0; i < frame_count; i++) {
    let p    = P[(i + 1) % m];
    let u1o1 = frames[i][0];
    let u2o2 = frames[i][1];

    let alpha = Alpha[(i + 1) % m];
    // u1 *= sign(alpha);
    // u2 *= sign(alpha);

    let b = mth.add(u1o1, u2o2);

    // Nasty sign-workaround to fix for almost opposite segments
    // #pragma warning TODO fix opposite / collinear workaround
    let ss1 = 1.;
    let ss2 = 1;
    // fix for nearly collinear or oppositely oriented segments
    let short_end_1 = false;
    let short_end_2 = false;
    // if (i==0 && norm(D.col(i)) < max(W.col(i))*2)
    //    short_end_1 = true;
    // if (i==m-2 && norm(D.col(i+1)) < max(W.col(i+1))*2)
    //    short_end_2 = true;

    // double fold_ang_amt = 2.*std::min(Math.abs(alpha), 1.) / ag::pi; //alpha /
    // (ag::pi/2); //std::min(Math.abs(alpha) / (ag::pi/2), 1.)*sign(alpha);
    let sigma        = params.fold_angle_sigma;
    let fold_ang_amt = 1. - Maht.exp(-(alpha * alpha) / (sigma * sigma));
    // std::cout << "fold_amt: " << sigma << std::endl;
    let unfold = params.unfold;

    let ip1 = (i + 1) % m;

    let u2_greater = mth.norm(u2o2) > mth.norm(D[ip1]);
    let u1_greater = mth.norm(u1o1) > mth.norm(D[i]);
    if (u1_greater ||  // norm(u1) > norm(D.col(i)) ||
        u2_greater ||  // norm(u2) > norm(D.col(i+1)) ||
        short_end_1 || short_end_2)

    {
      // use normals instead of local frame in that case
      // and use blended width
      let n  = mth.mul(mth.normalize(mth.add(N[ip1], N[i])), sign(alpha));
      unfold = true;
      let b      = mth.mul(n, Math.max(W[i][0], W[ip1, 0]));

      fold_ang_amt = 0;
      if (Math.abs(alpha) > Math.PI / 1.5) {
        // oppositely oriented apply sign workaround
        // this fixes the issue that with (almost) oppositely oriented sectors,
        // the normal will result in a self intersecting prototype poligon
        // so we consider the perpendicular to the bisector here
        // std::cout << "opposing sign" << std::endl;
        n   = mth.mul(mth.perp(mth.normalize(mth.add(u1o1, u2o2))), sign(alpha));
        b   = mth.mul(n, Math.max(W[i][0], W[ip1][0]));
        ss1 = -1;
        ss2 = 1;
      }
    }

// #if AG_DEBUG_DRAW
//     if (params.debug_draw) {
//       gfx::color(1, 0, 1);
//       gfx::drawLine(p, p + b);
//       gfx::drawLine(p, p - b);
//     }
// #endif

//     // Mitering
    let limit = Math.max(W(1, i), W(0, ip1)) * params.miter_limit;
    let    d1    = D[i];
    let    d2    = D[ip1];
    let    hu1   = mth.normalize(u1o1);
    let    hu2   = mth.normalize(u2o2);

    let    min_fold_length = Math.min(mth.norm(u1o1), mth.norm(u2o2));
    let concave_side;
    let convex_side;
    if (alpha < 0.) {
      concave_side = L;
      convex_side  = R;
    } else {
      concave_side = R;
      convex_side  = L;
    }

    let bb = apply_miter(b, p, d1, d2, limit);
    if (mth.has_nan(bb))
      console.log("SkeletalStroke: Mitering prudced NaN");
    convex_side.push(mth.addm(p, bb[0], ss1));
    convex_side.push(mth.addm(p, bb[1], ss2));

    if (unfold) {
      concave_side.push(mth.subm(p, b, ss1));
      concave_side.push(mth.subm(p, b, ss2));
    } else {
      concave_side.push(mth.add(mth.subm(p, b, ss1),
                                 mth.mul(hu1, min_fold_length * ss1 * fold_ang_amt)));
      concave_side.add(mth.add(mth.subm(p, b, ss2),
                                mth.mul(hu2, min_fold_length * ss2 * fold_ang_amt)));  //*fold_ang_amt*sign(alpha)*ss2);
    }
  }

  let    alpha = Alpha[0];
  let concave_side;
  let convex_side;
  if (alpha < 0.) {
    concave_side = L;
    convex_side  = R;
  } else {
    concave_side = R;
    convex_side  = L;
  }

  if (!closed) {
    L.add(mth.addm(P[P.lenght-1], N[N.length - 1], W[W.length - 1][1]));
    L.add(mth.subm(P[P.lenght-1], N[N.length - 1], W[W.length - 1][1]));
  } else {
    concave_side[0] = concave_side[concave_side.length-1];
    convex_side[0]  = convex_side[convex_side.length-1];
    concave_side.pop();
    convex_side.pop();
  }

  // Build prototype polygons
  let envelopes = [];

  for (let i = 0; i < L.size(); i += 2) {
    let proto =  [L[i], L[i + 1], R[i + 1], R[i]];
    if (debug){
      p5.noFill();
      p5.stroke(255,0,0);
      p5.polygon2(proto, true);
    }
    //if (params.debug)
    //  debug_draw_quad(proto);
    envelopes.push(proto);
  }

  // build homographies
  let homographies = [];
  let prototype_rects = [];

  let tl    = rect[0];
  let rd    = geom.rect_size(rect);
  let up    = [0, rd[1]];
  let right = [rd[0], 0];

  for (let i = 0; i < m; i++) {
    // source quad
    let t0 = xsub[i];
    let t1 = xsub[i + 1];

    let x0      = mth.addm( tl, right, t0 );
    let x1      = mth.addm( tl, right, t1 );
    let quad    = [x0, x1, mth.add(x1, up), mth.ad(x0, up)];
    prototype_rects.push(quad);
  }
}

module.exports = skeletal_strokes;
