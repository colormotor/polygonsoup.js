/**
 *  ______ _______ _____  ___ ___ _______ _______ _______ _______ _______ _______ ______          _____ _______
 * |   __ \       |     ||   |   |     __|       |    |  |     __|       |   |   |   __ \______ _|     |     __|
 * |    __/   -   |       \     /|    |  |   -   |       |__     |   -   |   |   |    __/______|       |__     |
 * |___|  |_______|_______||___| |_______|_______|__|____|_______|_______|_______|___|         |_______|_______|
 *
 * © Daniel Berio (@colormotor) 2021 - ...
 *
 * Minimal intervention control trajeectories.
 * Trajectory generation using Linear Quadratic Tracking (LQT) as described in:
 *    @inproceedings{BerioGI2017,
 *     author = {Berio, D and Calinon, S. and Fol Leymarie, F.},
 *     title = {Generating Calligraphic Trajectories with Model Predictive Control},
 *     booktitle = {Proceedings of Graphics Interface}, series = {GI 2017},
 *      year = {2017},
 *      address = {Edmonton, Canada},
 *      month = {May},
 *      publisher = {Canadian Human-Computer Communications Society},
 *   }
 *   @inproceedings{BerioMPCMOCO2017,
 *      author = {Berio, D and Calinon, S. and Fol Leymarie, F.},
 *      title = {Dynamic Graffiti Stylisation with Stochastic Optimal Control},
 *      booktitle = {ACM Proceedings of the 4th International Conference on Movement and Computing},
 *      series = {MOCO '17}, year = {2017}, month = {June}, address =
 *      {London, UK},
 *      publisher = {ACM}
 *   }
 *   @article{Berio2019,
 *     title = {Interactive Generation of Calligraphic Trajectories from Gaussian Mixtures},
 *     author = {Berio, Daniel and Fol Leymarie, Frederic and Calinon, Sylvain},
 *     publisher = {Springer}, journal = {Mixture Models and Applications},
 *     series = {Unsupervised and Semi-Supervised Learning},
 *     editor="Bouguila, N. and Fan, W.",
 *     pages = {23-38},
 *     year = {2019},
 *   }
 *   Please cite one of these papers if using this code for accademic purposes.
**/

"use strict";
const mth = require("./mth.js");
const discretization_default = 'exact';
const lqt = function() { };

const factorial =
  [1,
    1,
    2,
    6,
    24,
    120,
    720,
    5040,
    40320,
    362880,
    3628800,
    39916800,
    479001600,
    6227020800,
    87178291200,
    1307674368000,
    20922789888000,
    355687428096000,
    6402373705728000,
    121645100408832000];


lqt.discretize_system = (A, B, dt, method = discretization_default) => {
  if (mth.dim(B).length == 1)
    B = mth.transpose([B]);
  if (method == 'exact') {
    const dim = mth.dim(A)[0];
    var Ad = mth.zeros(dim, dim);
    for (var i = 0; i < dim; i++) {
      //mth.print(mth.diag(mth.ones(dim), i))
      Ad = mth.add(Ad, mth.mul(mth.shiftmat(dim, i), Math.pow(dt, i) * (1 / factorial[i])));
    }
    var Bd = mth.zeros(dim, 1);
    for (var i = 1; i < dim + 1; i++)
      Bd[dim - i][0] = Math.pow(dt, i) / factorial[i];
    return [Ad, Bd];
  } else if (method == 'zoh') {
    // var EM = mth.hstack([A, B]);
    // var zeros = mth.hstack([mth.zeros(mth.dim(B)[1], mth.dim(B)[0]),
    // mth.zeros(mth.dim(B)[1], mth.dim(B)[1])]);
    // EM = mth.vstack([EM, zeros]);
    // var M = mth.expm(mth.mul(EM, dt));
    // var Ad = mth.submat(M, [0, mth.dim(A)[0]], [0, mth.dim(A)[1]]);
    // var Bd = mth.submat(M, [0, mth.dim(B)[0]], [mth.dim(A)[1], mth.dim(M)[1]])
  } else { //Euler
    if (method != 'euler')
      console.log('Unknown discretization method ' + method)
    var Ad = mth.add(mth.identity(mth.dim(A)[0]),
      mth.mul(A, dt));
    var Bd = mth.mul(B, dt);
    return [Ad, Bd];
  }
}

lqt.make_cov_2d = (theta, scale) => {
  /// Builds a 2d covariance with a rotation and scale
  // rotation matrix
  var Phi = mth.eye(2);
  const ct = Math.cos(theta);
  const st = Math.sin(theta);
  Phi[0][0] = ct; Phi[0][1] = -st;
  Phi[1][0] = st; Phi[1][1] = ct;
  // scale matrix
  var S = mth.diag([scale[0] ** 2, scale[1] ** 2]);
  var cov = mth.dot(mth.dot(Phi, S), mth.transpose(Phi)); //, Phi.T);
  return cov;
}


lqt.SHM_r = (d, order, segment_duration) => {
  /// simple harmonic motion based scale factor,
  /// computes the control cost r in terms of maximum allowed displacement,
  /// assuming an oscillatory motion along a segment.
  /// transfer function is 1/s^k, order k, evaluated for j-omega and taking complex absolute value
  var omega = (Math.PI) / segment_duration; // Angular frequency
  // frequency gain, squared since R is a quadratic term
  return 1. / ((d * omega ** order) ** 2);
}


lqt.LinearSystem = function(order, subd, dim = 2, segment_duration = 0.5) {
  this.order = order;
  this.dim = dim;
  this.subd = subd;
  this.segment_duration = segment_duration;
  this.dt = segment_duration / subd;

  // (univariate) linear system definition x_dot = Ax + Bu
  var A = mth.zeros(order, order);
  mth.set_submat(A, [0, order - 1], [1, order], mth.eye(order - 1));
  this.A = A;
  var B = mth.zeros(order, 1);
  B[B.length - 1][0] = 1.;
  this.B = B;
  var C = mth.zeros(order, 1);
  C[0][0] = 1.0;
  this.C = mth.transpose(C);

  this.discretize = (dt = -1, method = discretization_default) => {
    if (dt <= 0)
      dt = this.dt;
    var [Ad, Bd] = lqt.discretize_system(this.A, this.B, dt, method);
    // extend to include all dims
    Ad = mth.kron(Ad, mth.eye(this.dim));
    Bd = mth.kron(Bd, mth.eye(this.dim));
    return [Ad, Bd];
  }
}

lqt.num_samples = (subd, m) => {
  return (m - 1) * subd + 1;
}

lqt.solve_batch = (sys, ref, d, opt_x0 = false, r = -1) => {
  /// Batch solution to LQT problem
  const k = sys.order;
  const dim = sys.dim;
  const dt = sys.dt;

  const cDim = dim * k; // # dimension of the full state

  var [MuQ, Q, K] = ref;
  const n = parseInt(mth.dim(MuQ)[0] / cDim); // cDim

  if (r <= 0)
    r = lqt.SHM_r(d, k, sys.segment_duration);

  var R = mth.mul(mth.eye(dim), r);
  R = mth.kron(mth.eye(n - 1), R);

  var [Ad, Bd] = sys.discretize();

  var Qs = [];
  var Sxs = [];
  var Sus = [];


  // Build Sx and Su matrices for batch LQR, see Eq. (14, 15)
  var Su = mth.zeros(cDim * n, dim * (n - 1))
  var Sx = mth.kron(mth.ones(n, 1), mth.eye(cDim));
  var M = Bd;//.slice(0); //mth.array(Bd)
  for (var i = 1; i < n; i++) {
    mth.set_submat(Sx, [i * cDim, n * cDim], [],
      mth.dot(mth.submat(Sx, [i * cDim, n * cDim], []), Ad));
    mth.set_submat(Su, [i * cDim, (i + 1) * cDim], [0, i * dim], M);
    M = mth.hstack([mth.dot(Ad,
      mth.submat(M, [], [0, dim])), M]);
  }



  var SuTQ = mth.dot(mth.transpose(Su), Q);
  var SxTQ = mth.dot(mth.transpose(Sx), Q);

  var pa = 0
  var pb = n - 1

  // # Solution
  var Uu = mth.add(mth.dot(SuTQ, Su), R); //  [SuTQi @ Su + R for SuTQi in SuTQs]) # U coefficients
  var Uxbar = mth.dot(SuTQ, MuQ); // SuTQi @ MuQi
  var X0bar = mth.dot(SxTQ, MuQ); // @ MuQi for SxTQi, MuQi in zip(SxTQs, MuQs)])

  const invf = mth.inv;

  if (!K.length) {
    if (opt_x0) {

      var Ux0 = mth.dot(SuTQ, Sx);
      var X0u = mth.dot(SxTQ, Su);
      var X0x0 = mth.dot(SxTQ, Sx);

      var Rq = mth.block([[Uu, Ux0],
      [X0u, X0x0]]);

      var rq = mth.vstack([Uxbar, X0bar]);
      var ux0 = mth.dot(invf(Rq), rq); // #mth.linalg.solve(Rq, rq)
      var uhat = ux0;
      var u = ux0.slice(0, mth.dim(Uxbar)[0]);
      var x0 = ux0.slice(mth.dim(Uxbar)[0], mth.dim(Uxbar)[0] + cDim);

      var uSigma = invf(Rq); // #; % + eye((nbData-1)*nbVarU) * 1E-10;
      var utox = mth.hstack([Su, Sx]);
      var xSigma = mth.dot(mth.dot(utox, uSigma), mth.transpose(utox));
    } else { // Fixed initial state

      var x0 = MuQ.slice(0, cDim); // .reshape(-1,1)

      var Sxx0 = mth.dot(Sx, x0);

      var Rq = mth.add(mth.dot(SuTQ, Su), R);
      var rq = mth.dot(SuTQ, mth.sub(MuQ, Sxx0));

      var u = mth.dot(invf(Rq), rq);
      var uhat = u;
      var uSigma = invf(Rq);
      var utox = Su;
      var xSigma = mth.dot(mth.dot(utox, uSigma), mth.transpose(utox));
    }
  } else { // periodic
    opt_x0 = true;

    var Ux0 = mth.dot(SuTQ, Sx);
    //Ux0 = SuTQ @ Sx
    var Ul = mth.dot(mth.transpose(Su), mth.transpose(K));
    var X0u = mth.dot(SxTQ, Su); //X0u = SxTQ @ Su
    var X0x0 = mth.dot(SxTQ, Sx);
    var X0l = mth.dot(mth.transpose(Sx), mth.transpose(K));
    var Lu = mth.dot(K, Su);
    var Lx0 = mth.dot(K, Sx);
    var Lxl = mth.zeros(cDim, cDim);
    var Lxl = mth.zeros(cDim, cDim);

    var X0bar = mth.dot(SxTQ, MuQ);

    var Rq = mth.block([[Uu, Ux0, Ul],
    [X0u, X0x0, X0l],
    [Lu, Lx0, Lxl]]);
    var rq = mth.vstack([Uxbar, X0bar, mth.zeros(cDim, 1)]);
    var ux0 = mth.dot(invf(Rq), rq); //#solve(Rq, rq) # #mth.linalg.solve(Rq, rq)
    var uhat = ux0;
    var u = ux0.slice(0, mth.dim(Uxbar)[0]);
    var x0 = ux0.slice(mth.dim(Uxbar)[0], mth.dim(Uxbar)[0] + cDim);

    var uSigma = invf(Rq); // #; % + eye((nbData-1)*nbVarU) * 1E-10;
    var utox = mth.hstack([Su, Sx, mth.transpose(K)]);
    var xSigma = mth.dot(mth.dot(utox, uSigma), mth.transpose(utox));
  }

  var res = {};

  var y = mth.add(mth.dot(Sx, x0), mth.dot(Su, u));

  res.x = mth.reshape(y, [n, cDim]);

  res.u = mth.reshape(u, [n - 1, dim]);
  res.uSigma = uSigma;
  res.xSigma = xSigma;
  res.sys = sys;
  res.x0 = x0;
  return res;
}

/// Resample a possibly sparse trajectory
lqt.resample = (res, subd = 1, observe = false, get_commands = false, discretization = 'exact') => {
  var x = res.x, u = res.u, sys = res.sys;
  var x0 = res.x[0];
  var n = mth.dim(x)[0];
  var dim = mth.dim(u)[1];
  var cDim = mth.dim(x)[1];

  var dt = sys.dt;
  var dts = dt / (subd);

  var ns = lqt.num_samples(subd, n);
  var xs = mth.zeros(cDim, ns);
  var us = mth.zeros(dim, ns - 1);

  var [Ad, Bd] = sys.discretize(dts, discretization);

  var t = 0;
  xs[0] = x[0];
  for (var i = 0; i < n - 1; i++) {
    //xs[t] = x[i];
    for (var j = 0; j < subd; j++) {
      xs[t + 1] = mth.add(mth.dot(Ad, xs[t]), mth.dot(Bd, u[i])); // u[:,i]
      us[t] = u[i];
      t += 1;
    }
  }

  if (get_commands)
    return [xs, us];
  return xs
}

const chord_lengths = (P, closed = 0) => {
  var D = mth.diff(P, 0);
  return D.map((d) => Math.sqrt(d[0] ** 2 + d[1] ** 2));
}

const cum_chord_lengths = (P, closed = 0) => {
  var L = chord_lengths(P, closed);
  return mth.cumsum([...[0.0], ...L]);
}

const centripetal_parameters = (P, m, n, alpha) => {
  var L = chord_lengths(P);
  L = mth.pow(L, alpha);
  L = mth.cumsum([...[0.0], ...L]);
  L = mth.div(L, L[L.length - 1]);
  L[L.length - 1] = 1.0;
  return mth.mul(L, n - 1);
}

lqt.sigma_rbf = (subd, activation_sigma) => {
  // Activation function for variable support tracking reference
  // (half) Full width at half maximum of Gaussian
  return Math.max(1e-5, (subd * 0.5 * activation_sigma) / 3);
}

const rbf = (x, mu, sigma) => {
  var sq = mth.pow(mth.sub(x, mu), 2);
  return mth.exp(mth.div(sq, -(2 * sigma ** 2)));
}

lqt.reference_activations = (P, m, n, activation_sigma, alpha = 0., get_activations = false, psg = null, get_ref = false) => {
  var subd = (n - 1); //(m - 1)
  var sigma = lqt.sigma_rbf(subd, activation_sigma);

  var tau = centripetal_parameters(P, P.length, n, alpha);

  // activations
  var H = mth.zeros(m, n);
  var t = mth.linspace(0, n - 1, n);
  var hrow = mth.ones(n);
  for (var i = 0; i < tau.length; i++) {
    var tau_i = tau[i];
    var p = rbf(t, parseInt(tau_i), sigma); //#mth.random.uniform(2, 130)) #(n/m)*0.01)

    H[i] = p; // mth.mul(hrow,  p);
  }


  P = H;
  H = mth.div(H, (mth.repmat(mth.add(mth.sum(H, 0), Math.exp(-6)), m, 1)));

  var I = [];
  var T = [];
  var v = mth.mul(mth.ones(n), -1);

  for (var t = 0; t < n; t++)
    for (var i = 0; i < H.length; i++)
      if (H[i][t] > 0.5) {
        I.push(i);
        T.push(t);
        v[t] = i;
        break
      }

  if (get_ref)
    return v.map((vi) => parseInt(vi));
  if (get_activations)
    return [P, H];
  return [I, T];
}


/// Reference trajectory in cartesian coordinates
lqt.cartesian_reference = (sys, P, Sigma, activation_sigma = 0.5,
  periodic = false,
  stop_movement = true,
  alpha = 0.0,
  clamp = false,
  diag = 0) => {
  if (periodic) {
    P = mth.vstack([P, [P[0]]]);
    Sigma = [...Sigma, ...[Sigma[0]]];
    stop_movement = false;
  } else if (clamp) {
    P = mth.vstack([[P[0]], P, [P[P.length - 1]]]);
    Sigma = [...[Sigma[0]], ...Sigma, ...[Sigma[Sigma.length - 1]]];
    // stop_movement = False
  }

  var [m, pdim] = mth.dim(P);
  var dim = sys.dim;
  var order = sys.order;
  var segment_duration = sys.segment_duration;
  var subd = sys.subd;
  var cDim = dim * order

  // number of samples:
  var n = lqt.num_samples(subd, m);


  // full precisions
  var Lambda = [];
  for (var i = 0; i < m; i++) {
    var l = mth.eyev(cDim, diag);
    if (mth.max(mth.abs(Sigma[i])) > 1e-10)
      mth.set_submat(l, [0, pdim], [0, pdim], mth.inv(Sigma[i]));
    Lambda.push(l);
  }

  var MuQ = _.range(0, n).map((v) => mth.zeros(cDim));
  var Q = _.range(0, n).map((v) => mth.zeros(cDim, cDim));

  // state activations
  var [states, timesteps] = lqt.reference_activations(P, m, n, activation_sigma, alpha);

  // pad keypoints
  var Mu = mth.hstack([P, mth.zeros(m, cDim - pdim)]);
  for (var i = 0; i < states.length; i++) {
    MuQ[timesteps[i]] = Mu[states[i]];
    Q[timesteps[i]] = Lambda[states[i]];
  }

  MuQ = mth.transpose([mth.concatenate(MuQ)]);
  Q = mth.block_diag(Q);

  var pa = 0, pb = n - 1;
  if (periodic) {
    var K = mth.zeros(cDim, n * cDim)
    mth.set_submat(K, [], [pa * cDim, pa * cDim + cDim], mth.eyev(cDim, 1));
    mth.set_submat(K, [], [pb * cDim, pb * cDim + cDim], mth.eyev(cDim, -1));
  } else {
    var K = [];
    if (stop_movement) {
      // # TODO add constraint here
      // # i = (n-1)*cDim
      // # K = mth.zeros((cDim, n*cDim))
      // # K[:, i:i+cDim] = mth.diag(mth.concatenate([mth.zeros(dim), mth.ones(cDim-dim)]))
      // # print('stopit')
      var i = (n - 1) * cDim;
      // Q[i+dim:i+cDim,i+dim:i+cDim] = mth.eye(cDim-dim)
      // Q[i:i+dim,i:i+dim] = mth.eye(dim)*1 #100
      mth.set_submat(Q, [i + dim, i + cDim], [i + dim, i + cDim], mth.eye(cDim - dim));
      mth.set_submat(Q, [i, i + dim], [i, i + dim], mth.eyev(dim, 1));
    }
  }

  return [MuQ, Q, K];
}

/// Create LQT trajectory with tied covariances
lqt.tied_lqt = function(
  P,
  params = {}) {
  if (params.angle == undefined) params.angle = 30;
  if (params.isotropy == undefined) params.isotropy = 0.5;
  if (params.sigma == undefined) params.sigma = 3;
  if (params.activation_sigma == undefined) params.activation_sigma = 0.5;
  if (params.d == undefined) params.d = 5;
  if (params.order == undefined) params.order = 4;
  if (params.segment_duration == undefined) params.segment_duration = 0.5;
  if (params.opt_x0 == undefined) params.opt_x0 = true;
  if (params.stop_movement == undefined) params.stop_movement = true;
  if (params.periodic == undefined) params.periodic = true;
  if (params.subd == undefined) params.subd = 5;
  if (params.subsample == undefined) params.subsample = 4;
  if (params.dim == undefined) params.dim = 2;
  var Sigma = [];
  var cov = mth.mul(mth.eye(params.dim), params.sigma ** 2);
  mth.set_submat(cov, [0, 2], [0, 2], lqt.make_cov_2d(mth.radians(params.angle), [params.isotropy * params.sigma, params.sigma]));
  for (const p of P)
    Sigma.push(cov);

  var sys = new lqt.LinearSystem(params.order, params.subd, params.dim, params.segment_duration);

  var ref = lqt.cartesian_reference(sys, P, Sigma, params.activation_sigma, params.periodic, params.stop_movement);
  var res = lqt.solve_batch(sys, ref, params.d, params.opt_x0);

  if (params.subsample > 0)
    var x = lqt.resample(res, params.subsample);
  else
    var x = res.x;
  return x;
}

lqt.evolute = (X, get_radii = true) => {
  X = mth.transpose(X);

  var [x, y] = X.slice(0, 2);
  var [dx, dy] = X.slice(2, 4);
  var [ddx, ddy] = X.slice(4, 6);

  var dxy2 = mth.add(mth.mul(dx, dx), mth.mul(dy, dy));
  var dddxy = mth.sub(mth.mul(dx, ddy), mth.mul(ddx, dy));

  var r = mth.div(mth.pow(dxy2, 3 / 2),
    mth.abs(dddxy));

  x = mth.sub(x, mth.div(mth.mul(dxy2, dy), dddxy));
  y = mth.add(y, mth.div(mth.mul(dxy2, dx), dddxy));

  return [mth.transpose([x, y]), r];
}

lqt.test = () => {
  var A = [[0, 1], [1, 0]];
  // console.log(mth.factorial(3));//(mth.ones(4), 1));//size(mth.transpose(A)));
  // console.log(lqt.make_cov_2d(3.0, [0.5, 1.0]));
  // //console.log(mth.kron(mth.eye(2), [1,1]));
  // console.log(mth.hstack([mth.eye(2), mth.ones(2, 3)]));
  // console.log(mth.sum(mth.ones(3, 2)));
  // console.log(mth.transpose(mth.ones(3)));
  // console.log(mth.zeros(4,1))
  // var m = mth.set_submat(mth.zeros(5,5), [2,4], [2,4], [[5, 1], [2, 3]]);//, mth.eye(2) ));
  // mth.print(m)
  // mth.print(mth.transpose(mth.submat(m, [2, 4], [2, 4])))
  // mth.print(mth.block_diag([mth.eyev(2, 1), mth.eyev(2, 2)]))
  var sys = new lqt.LinearSystem(4, 5)
  var [Ad, Bd] = sys.discretize()
  mth.print(mth.shiftmat(4, -2))
  //mth.print(mth.set_submat(mth.zeros(5,5), [0,3], [0,3], mth.eye(3)));//, mth.eye(2) ));
}

module.exports = lqt;
