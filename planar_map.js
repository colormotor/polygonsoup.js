"use strict";
const DCEL = require('dcel.js')
const mth = require('./mth.js')
const geom = require('./geom.js')

import sweepline from "./sweepline.js";
const planar_map = function() { }


planar_map.compute_faces = (segs, get_intersections = false, debug_ind = -1, avoid_incident=true) => {
  let res = sweepline.sweepline_intersections(segs, avoid_incident, debug_ind);
  let faces = [];


  let dcel = new DCEL.DCEL(res.graph.vertices, res.graph.edges);
  let dfaces = dcel.internalFaces();
  for (let face of dfaces)
      faces.push(face.vertexlist.map(v => [v.x, v.y]))

  if (get_intersections)
    return [faces, res];
  return faces;
}


planar_map.compute_shape_faces = (S, closed=false, get_intersections = false, debug_ind = -1) => {
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
  try{
    return planar_map.compute_faces(segs, get_intersections, debug_ind, false);
  }catch(err){
    console.log('DCEL error while computing faces');
    let verts = [];
    let edges = [];
    let i = 0;
    for (const P of S){
      verts = verts.concat(P);
      edges = edges.concat(geom.edges(P, closed).map(e=>[e[0]+i, e[1]+i]));
      i += P.length;
    }
    let dcel = new DCEL.DCEL(verts, edges);
    let dfaces = dcel.internalFaces();
    let faces = [];
    for (let face of dfaces)
      faces.push(face.vertexlist.map(v => [v.x, v.y]))
    if (get_intersections)
      return [faces, null];
    return faces;
  }
}



//module.exports = planar_map
export default planar_map;
