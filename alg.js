/**
 *  ______ _______ _____  ___ ___ _______ _______ _______ _______ _______ _______ ______          _____ _______
 * |   __ \       |     ||   |   |     __|       |    |  |     __|       |   |   |   __ \______ _|     |     __|
 * |    __/   -   |       \     /|    |  |   -   |       |__     |   -   |   |   |    __/______|       |__     |
 * |___|  |_______|_______||___| |_______|_______|__|____|_______|_______|_______|___|         |_______|_______|
 *
 * © Daniel Berio (@colormotor) 2021 - ...
 *
 * some handy datstructures and algorithms
 **/

'use strict';
const {MinQueue} = require("heapify");

const alg = function() { }

/**
 * A^* path finding.
 * @param {any} G graph (assumed to be graphology graph https://github.com/graphology/graphology)
 * @param {any} start
 * @param {any} goal
 * @param {any} dist
 * @param {any} heuristic
 * @returns
 */
alg.astar = (G, start, goal, dist=(n)=>1, heuristic=(n)=>0) => {
  let openSet = new MinQueue(G.order*3);
  openSet.push(start, 0);

  let cameFrom = {};

  let gScore = {};
  gScore[start] = 0;
  let fScore = {};
  fScore[start] = heuristic(start);

  const reconstruct_path = (current) => {
    let total_path = [current];
    while (current in cameFrom){
      current = cameFrom[current];
      total_path.push(current);
    }
    return total_path.reverse();
  };

  while (openSet.size) {
    let current = openSet.pop();
    if (current == goal)
      return reconstruct_path(current);

    for (const bor of G.neighbors(current)){
      const s = gScore[current] + dist(current, bor);
      if (! (bor in gScore) || s < gScore[bor]){
        cameFrom[bor] = current;
        gScore[bor] = s;
        fScore[bor] = s + heuristic(bor);
        openSet.push(bor, fScore[bor]);
      }
    }
  }
  return null;
}


alg.DoublyLinkedList = function() {
  this.front = null;
  this.back = null;

  this.clear = () => {
    this.front = null;
    this.front = null;
  }

  this.print = () => {
    console.log("Hello");
  }

  this.add = (elem, at = null) => {
    elem.prev = null;
    elem.next = null;
    if (this.back == null) {
      this.back = this.front = elem;
      return
    }
    if (at == null) {
      at = this.back;
      this.back = elem;
      elem.next = null;
    } else {
      elem.next = at.next;
    }
    elem.prev = at;
    at.next = elem;
  }

  this.insert = (elem, at = null) => {
    elem.prev = null;
    elem.next = null;
    if (this.back == null) {
      this.back = this.front = elem;
      return
    }
    if (at == null) {
      at = this.front;
      this.back = elem;
      elem.prev = null;
    } else {
      elem.prev = at.prev;
    }

    elem.next = at;
    at.prev = elem;
  }

  this.nodes = function*() {
    let e = this.front;
    while (e != null) {
      yield (e);
      e = e.next;
    }
  }

  this.pop_front = () => {
    if (this.front != null) {
      let res = this.front;
      if (this.front == this.back) {
        this.clear()
      } else {
        this.front.next.prev = null;
        this.front = this.front.next;
      }
      res.prev = res.next = null;
      return res;
    } else {
      return null;
    }
  }

  this.pop_back = () => {
    if (this.back != null) {
      let res = this.back;
      if (this.front == this.back) {
        this.clear();
      } else {
        this.back.prev.next = null;
        this.back = this.back.prev;
      }
      res.prev = res.next = null;
      return res;
    } else {
      return null;
    }
  }

  this.remove = (elem) => {
    if (elem == this.front)
      return this.pop_front();
    if (elem == this.back)
      return this.pop_back();
    // should have prev and next since it isnt front or back
    if (elem.next != null)
      elem.next.prev = elem.prev;

    if (elem.prev != null)
      elem.prev.next = elem.next;

    elem.prev = elem.next = null;
    return elem;
  }
}

alg.DoublyLinkedList.Node = function() {
  this.prev = null;
  this.next = null;
}

module.exports = alg;
