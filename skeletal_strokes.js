/**
 *  ______ _______ _____  ___ ___ _______ _______ _______ _______ _______ _______ ______          _____ _______
 * |   __ \       |     ||   |   |     __|       |    |  |     __|       |   |   |   __ \______ _|     |     __|
 * |    __/   -   |       \     /|    |  |   -   |       |__     |   -   |   |   |    __/______|       |__     |
 * |___|  |_______|_______||___| |_______|_______|__|____|_______|_______|_______|___|         |_______|_______|
 *
 * © Daniel Berio (@colormotor) 2021 - ...
 *
 * Polygonal skeletal strokes based on
 *
 * @inproceedings{BerioExpressive2019,
 *  author = {Berio, Daniel and Asente, Paul and Echevarria, Jose and Fol Leymarie, Frederic},
 *  title = {Sketching and Layering Graffiti Primitives},
 *  year = {2019},
 *  publisher = {Eurographics Association},
 *  address = {Goslar, DEU},
 *  doi = {10.2312/exp.20191076},
 *  booktitle = {Proceedings of the 8th ACM/Eurographics Expressive Symposium on Computational Aesthetics and Sketch Based Interfaces and Modeling and Non-Photorealistic Animation and Rendering},
 *  pages = {51â€“59},
 *  numpages = {9},
 *  location = {Genoa, Italy},
 *  series = {Expressive '19}
 *
 *  }
 **/

"use strict"
const alg = require("./alg.js");
const geom = require("./geom.js");
const mth = require("./mth.js");

const skeletal_strokes = function() {}

/* Fuzzy eqs */
const zero_eps = 1e-5;
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
                     partition_indices = [],
                     spine_index = -1,
                     flipped = false) => {
                       return {pos: pos,
                               type: type,
                               partition_index: partition_index,
                               partition_indices: partition_indices,
                               spine_index: spine_index,
                               flipped: flipped,
                               loop_point: false,
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
skeletal_strokes.split_polyline_horizontal = (P,
                                              closed,
                                              spine_closed,
                                              xs,
                                              flip_normals,
                                              duplicate_joints, p5=null) =>
{
  let nsign = 1;
  let edges = geom.edges(P, closed);

  let N = [];
  if (flip_normals) nsign = -1;
  for (let i = 0; i < edges.length; i++) {
    let e = edges[i];
    let  a = P[e[0]];
    let  b = P[e[1]];
    if (mth.distance(a, b) < zero_eps) {
      if (i < edges.length - 1)
        b = P[edges[i + 1][1]];
      else
        a = P[edges[i - 1][1]];
    }
    N.push(mth.mul(mth.perp(mth.normalize(mth.sub(b, a))), nsign));
  }

  //let N = geom.normals_2d(P, closed, false);

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

  ET.sort((a,b)=>compare_nums(P[a.a][0], P[b.a][0]));

  let AET = [];  // active edge table

  let flesh_data = [];
  let vlist = new alg.DoublyLinkedList();
  for (let i = 0; i < P.length; i++) {
    let v = FleshVertex(P[i]);
    if (fequal(v.pos[0], 0)){
      v.type = "START";
    }
    flesh_data.push(v);

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

  let m;
  if (spine_closed){
    xs_sorted = xs_sorted.slice(1, xs.length);
    m = xs_sorted.length;
  }else{
    xs_sorted = xs_sorted.slice(1, xs.length-1);
    m = xs_sorted.length;
  }
  // console.log(m);
  const CLAMP_PARTITION = (i) => (spine_closed)?mth.imod(i, m):parseInt(Math.max(Math.min(i, m), 0));

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

    // sort AET according to second point
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
          flesh_data[e.a].partition_indices = [CLAMP_PARTITION(i)];
          flesh_data[e.a].flipped = e.flipped;
        }
        has_intersection = false;

      } else if (fequal(P[e.b][0], x)) {
        flesh_data[e.b].type = "JOINT";
        flesh_data[e.b].pos[0] = x;
        if (flesh_data[e.a].partition_index == -1) {
          flesh_data[e.a].partition_index = CLAMP_PARTITION(i);
          flesh_data[e.a].partition_indices = [CLAMP_PARTITION(i)];
          flesh_data[e.a].flipped = e.flipped;
        }
        has_intersection = false;
      }

      // we may have passed this segment all together,
      // so also in this case there will be no intersection
      if (fleq(P[e.b][0], x)) {
        if (flesh_data[e.b].partition_index == -1) {
          flesh_data[e.b].partition_index = CLAMP_PARTITION(i);
          flesh_data[e.b].partition_indices = [CLAMP_PARTITION(i)];
          flesh_data[e.b].flipped = e.flipped;
        }
        has_intersection = false;
      }

      // setup info for the first edge vertex the first time it is encountered
      if (fleq(P[e.a][0], x)) {
        // make sure partition are set
        if (flesh_data[e.a].partition_index == -1) {
          flesh_data[e.a].partition_index = CLAMP_PARTITION(i);
          flesh_data[e.a].partition_indices = [CLAMP_PARTITION(i)];
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
                            [CLAMP_PARTITION(i)],
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
  }

  // make sure we did not miss the final vertices
  let v = vlist.front;
  while (v) {
    if (v.partition_index == -1) {
      if (!spine_closed){
        v.partition_index = m;
        v.partition_indices = [m];
      }else{
        v.partition_index = m-1;
        v.partition_indices = [m-1];
      }
    }

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
          const old_ind = v.partition_index;
          v.partition_index = CLAMP_PARTITION(v.partition_index + 1);

          if (v.partition_index - old_ind < 0){
            v.loop_point = true;
            //console.log(v.partition_indices);
          }else{
            v.partition_indices = [old_ind, v.partition_index];
            vdup.partition_indices = [old_ind, v.partition_index];
          }

        } else {
          const old_ind = vdup.partition_index;
          vdup.partition_index = CLAMP_PARTITION(vdup.partition_index + 1);

          if (vdup.partition_index - old_ind < 0){
            vdup.loop_point = true;
            //console.log(v.partition_indices);
          }else{
            vdup.partition_indices = [old_ind, vdup.partition_index];
          v.partition_indices = [old_ind, vdup.partition_index];
          }
          //console.log(v.partition_indices);
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
  if (params.debug!=undefined){
    debug=true;
    p5 = params.debug;
  }
  let compound = true;
  if (!geom.is_compound(prototype)){
    prototype = [prototype];
    compound = false;
  }
  if (rect==null)
    rect = geom.bounding_box(prototype);

  const proto_box = geom.bounding_box(prototype);

  const aspect_l = Math.abs(geom.rect_l(rect) - geom.rect_l(proto_box)) / geom.rect_h(rect);
  const aspect_r = Math.abs(geom.rect_r(proto_box) - geom.rect_r(rect)) / geom.rect_h(rect);

  // if (closed){
  //   spine = spine.concat(spine.slice(0,2));
  //   width_profile = width_profile.concat(width_profile.slice(0,2)); //[W[0]]);
  // }

  // normalize control path and widths
  const    ch_lengths      = geom.cum_chord_lengths(spine, closed);
  const    segment_lengths = geom.chord_lengths(spine, closed);
  const l               = geom.chord_length(spine, closed);
  if (l < zero_eps)
    return;

  let P = spine;
  let W = width_profile;

  if (mth.dim(width_profile).length < 2)
    W = _.range(0, W.length-1).map(i=>[W[i], W[i+1]]);
  W = mth.mul(W, geom.rect_h(rect));

  //const edges = geom.edges(P, closed); // TODO closed

  // normals, bisectors and other values
  let D = geom.tangents(P, closed);
  let N = D.map(d=>mth.neg(mth.perp(mth.normalize(d))));

  let Alpha = geom.turning_angles(P, closed, true);  // P.closed, true);

  // for (let i = 0; i < D.length; i++){
  //   const l = mth.norm(D[i]);
  //   W[i][0] = Math.min(W[i][0], l/6);
  //   W[i][1] = Math.min(W[i][1], l/6);
  // }
  // Almost flat corners break with stepwise profile
  // TODO Either find a better solution or adjust based on width/angle
  let alpha_range;
  if (closed)
    alpha_range = _.range(0, Alpha.length);
  else
    alpha_range = _.range(1, W.length);
  // for (const i of alpha_range) {
  //   if (Math.abs(Alpha[i]) < mth.radians(20)){
  //     W[mth.imod(i-1, W.length)][1] = W[i][0];
  //   }
  // }
  //console.log([D.length, W.length, Alpha.length])
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

    if (isNaN(o1) || isNaN(o2)){
      console.log("sin(alpha)==0");
    }
    let u1 = mth.mul(d1, o1 * Math.sign(alpha));  // eq 1
    let u2 = mth.mul(d2, -o2 * Math.sign(alpha));

    if (debug){
    p5.stroke(0,255,0);
    p5.circle(...p, 10);
    p5.line(...p, ...mth.add(p, u1));
    p5.stroke(0,0,255);
    p5.line(...p, ...mth.add(p, u2));
    }

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
    let ss1 = 1;
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
    let fold_ang_amt = 1. - Math.exp(-(alpha * alpha) / (sigma * sigma));
    //let fold_ang_amt = 1-Math.sin(Math.abs(alpha)/2);
    // std::cout << "fold_amt: " << sigma << std::endl;
    let unfold = params.unfold;

    let ip1 = (i + 1) % m;

    let u2_greater = mth.norm(u2o2) > mth.norm(D[ip1]);
    let u1_greater = mth.norm(u1o1) > mth.norm(D[i]);
    if (u1_greater ||
        u2_greater ||
        short_end_1 || short_end_2)
    //if (false)
    {
      // use normals instead of local frame in that case
      // and use blended width
      let n  = mth.mul(mth.normalize(mth.add(N[ip1], N[i])), Math.sign(alpha));

      unfold = true;
      const maxl = 100;
      if (mth.norm(b) > maxl)
        b      = mth.mul(n, Math.max(W[i][0], W[ip1][0]));
      //b = mth.mul(mth.normalize(b), maxl);
        //console.log('problem');
      // b      = mth.mul(n, Math.max(W[i][0], W[ip1][0]));
      fold_ang_amt = 0;
      if (Math.abs(alpha) > Math.PI / 1.5) {
        // oppositely oriented apply sign workaround
        // this fixes the issue that with (almost) oppositely oriented sectors,
        // the normal will result in a self intersecting prototype poligon
        // so we consider the perpendicular to the bisector here
        // std::cout << "opposing sign" << std::endl;
        n   = mth.mul(mth.perp(mth.normalize(mth.add(u1o1, u2o2))), Math.sign(alpha));
        b   = mth.mul(n, Math.max(W[i][0], W[ip1][0]));
        
        ss1 = -1;
        ss2 = 1;
        
      }
    }

     if (debug){
      p5.noFill();
      p5.stroke(0,255,0);
      p5.line(...p, ...mth.add(p, b));
      p5.line(...p, ...mth.sub(p, b));

     }

// #if AG_DEBUG_DRAW
//     if (params.debug_draw) {
//       gfx::color(1, 0, 1);
//       gfx::drawLine(p, p + b);
//       gfx::drawLine(p, p - b);
//     }
// #endif
   
//     // Mitering
    let limit = Math.max(W[i][1], W[ip1][0]) * params.miter_limit;
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
      concave_side.push(mth.add(mth.subm(p, b, ss2),
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
    L.push(mth.addm(P[P.length-1], N[N.length - 1], W[W.length - 1][1]));
    R.push(mth.subm(P[P.length-1], N[N.length - 1], W[W.length - 1][1]));
  } else {
    concave_side[0] = concave_side[concave_side.length-1];
    convex_side[0]  = convex_side[convex_side.length-1];
    concave_side.pop();
    convex_side.pop();
  }

  // Build prototype polygons and envelopes
  let envelopes = [];

  for (let i = 0; i < L.length; i += 2) {
    let proto =  [L[i], L[i + 1], R[i + 1], R[i]];
    if (debug){
      // p5.noFill();
      // p5.stroke(255,0,0);
      // p5.polygon2(proto, true);
    }
    envelopes.push(proto);
  }
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
    let quad    = [x0, x1, mth.add(x1, up), mth.add(x0, up)];
    prototype_rects.push(quad);
  }

  const norm_rec = NormalizeRec(rect);

  // compute subdivision points
  let flip_normals = false;
  let alternate_normals = false;
  let res = {
    closed:closed,
    spine: spine,
    flesh: [],
    flesh_vertices: [] };

  for (const F_ of prototype) {
    let F = norm_rec.normalize_poly(F_); //normalize_polyline(F_, rect);
    //console.log(xsub);
    let vlist = skeletal_strokes.split_polyline_horizontal(F,
                                                           true, //TODO fixme
                                                           closed,
                                          xsub,
                                          flip_normals,
                                          params.duplicate_joints);
    if (alternate_normals) flip_normals = !flip_normals;

    let num_verts = 0;
    let vi        = 0;
    for (let v of vlist) {
      let pind = v.partition_index; //Math.min(v.partition_index, P.length - 1);
      // Project position according to prototype
      //mth.print(undefined);
      let pos = norm_rec.unnormalize(v.pos);

      let aspect   = norm_rec.w / l;
      let    proj_pos = transform_with_envelope(
        pind,
        v.loop_point,
          pos,
          prototype_rects,
          envelopes,
          l,
          aspect_l,
          aspect_r);
      v.edge_normal = mth.normalize(mth.sub(transform_with_envelope(pind, v.loop_point,
                                                            mth.add(pos, v.edge_normal),
                                                            prototype_rects,
                                                            envelopes,
                                                            aspect,
                                                            aspect_l,
                                                            aspect_r),
                                            proj_pos));
      if (mth.has_nan(proj_pos)){
        mth.print(undefined);
        console.log("proj_pos Nan");
      }
      v.pos = proj_pos;

      if (vi == 0)  // || vi==vlist.size()-1)
        v.spine_index = 0;

      if (vi == vlist.length - 1) {
        if (closed) {
          v.spine_index = 0;
        } else {
          v.spine_index = spine.length - 1;
        }
      }

      // check for joint type (concave or convex)
      let is_joint = v.type == "JOINT";
      // if (closed) {
      //   is_joint = is_joint || (vi == 0 || vi == vlist.length - 1);
      // }

      if (is_joint) {
        let th = Alpha[v.spine_index];
        let d  = mth.add(mth.normalize(mth.neg(D[mth.imod(v.spine_index - 1, m)])),
                         mth.normalize(D[mth.imod(v.spine_index, m)]));

        if (v.loop_point)
        {
          v.partition_index = envelopes.length-1;
          v.partition_indices = [v.partition_index];
        }

        if (mth.dot(d, v.edge_normal) > 0) {
          v.concave = true;
        } else {
        }
      } else {
        // set spine index according to projection over segment realtive postion
        let    pa = spine[pind];
        let    pb = spine[(pind + 1) % m];
        let d = mth.sub(pa, pb);
        let dot = mth.dot(mth.sub(v.pos, pa), d);

        let denom = mth.dot(d, d);
        let t  = geom.project(v.pos, pa, pb);
        if (t < 0.5)
          v.spine_index = pind;
        else
          v.spine_index = (pind + 1) % (m + 1);


        if (vi == 0) v.spine_index = 0;
        if (vi == vlist.length - 1) v.spine_index = spine.length - 1;
      }
      num_verts++;
      vi++;
    }  // end vertex loop

    //mth.print(undefined);
    res.flesh_vertices.push(vlist);
    res.flesh.push(vlist.map(v=>v.pos));
  }

  if (!compound){
    res.flesh = res.flesh[0];
    res.flesh_vertices = res.flesh_vertices[0];
  }
  return res;
}

const apply_miter = (b, p, d1, d2, limit) => {
  let l = mth.norm(b);

  if (l <= limit)
    return [b, b];

  let bu = mth.div(b, l);
  let bp = mth.mul(mth.perp(bu), 10);

  let p1a = mth.add(p, b);
  let p1b = mth.sub(p1a, d1);

  let p2a = mth.add(p, b);
  let p2b = mth.add(p2a, d2);

  let bp1 = mth.zeros(2);
  let bp2 = mth.zeros(2);
  let res;
  [res, bp1] = geom.line_segment_intersection(
    mth.addm(p, bu, limit),
    mth.add(mth.addm(p, bu, limit), bp),
                            p1a,
                            p1b);
  [res, bp2] = geom.line_segment_intersection(
    mth.addm(p, bu, limit),
    mth.add(mth.addm(p, bu, limit), bp),
                            p2a,
                            p2b);

  const bb = [mth.sub(bp1, p), mth.sub(bp2, p)];
  return bb;
}

const envelope_origin = (env) => {
  return mth.div(mth.add(env[0], env[3]), 2);
};

const envelope_side = (env) => {
  return mth.sub(mth.div(mth.add(env[1], env[2]), 2),
                 mth.div(mth.add(env[0], env[3]), 2));
};

const envelope_up = (env, t) => {
  let u1 = mth.sub(env[3], env[0]);
  let u2 = mth.sub(env[2], env[1]);
  return mth.addm(u1, mth.sub(u2, u1), t);
};

const envelope_size = (env, t) => {
  let side = envelope_side(env);
  let up   = envelope_up(env, t);
  return [mth.norm(side), mth.norm(up)];
}

const transform_with_envelope = (pind,
                                 loop_point,
                                 p,
                                 prototype_rects,
                                 envelopes,
                                 l,
                                 aspect_l,
                                 aspect_r) => {
  let src = prototype_rects[pind];
                                   let dst = envelopes[pind];
                                   if (loop_point)
                                     src = prototype_rects[prototype_rects.length-1];
                                   let q = p;
  p = mth.div(mth.sub(p, envelope_origin(src)), envelope_size(src, 0.));
  let t             = p[0];
  let tclip         = mth.clamp(t, 0.0, 1.0);
                                   if (loop_point)
                                     tclip = 0.0;
  let w = mth.norm(envelope_side(dst));
  let h = mth.norm(envelope_up(dst, tclip));
  let a = (h / l);
  if (t < 0.0) {
    t *= a * aspect_l;
  } else if (t > 1.0) {
    t = 1. + (t - 1) * a * aspect_r;
  }

    //                               console.log('cazz');
  //mth.print(undefined);
  return mth.add(mth.addm(envelope_origin(dst), envelope_side(dst), tclip),
                                                  mth.mul(envelope_up(dst, tclip), p[1]));
}
// Generates shape normalization info for skeletal strokes
const NormalizeRec = (rect) => {
  let    bmin = rect[0];
  let    bmax = rect[1];
  let y    = mth.addm(bmax[1], bmin[1], 0.5);
  let x0    = [bmin[0], y];
  let w     = bmax[0] - bmin[0];

  return {x0:x0, w:w,
          normalize: (p) => mth.div(mth.sub(p, x0), w),
          unnormalize: (p) => mth.madd(p, w, x0),
          normalize_poly: (P) => P.map(p=>mth.div(mth.sub(p, x0), w)),
          unnormalize_poly: (P) => P.map(p=>mth.madd(p, w, x0))
         };
};

module.exports = skeletal_strokes;
