/**
 *  ______ _______ _____  ___ ___ _______ _______ _______ _______ _______ _______ ______          _____ _______
 * |   __ \       |     ||   |   |     __|       |    |  |     __|       |   |   |   __ \______ _|     |     __|
 * |    __/   -   |       \     /|    |  |   -   |       |__     |   -   |   |   |    __/______|       |__     |
 * |___|  |_______|_______||___| |_______|_______|__|____|_______|_______|_______|___|         |_______|_______|
 *
 * © Daniel Berio (@colormotor) 2021 - ...
 *
 * wrappper around p5js
 *
 **/

'use strict';
var p5 = require("p5");
var mth = require("./mth.js");

p5.prototype.polygon2 = function(P, closed=false){
  if (this.drawingContext instanceof WebGLRenderingContext)
    this.beginShape(this.TESS);
  else
    this.beginShape();
  for (const p of P)
    this.vertex(p[0], p[1]);
  if (closed)
    this.endShape(this.CLOSE);
  else
    this.endShape();
};

p5.prototype.setLineDash = function(list) {
  this.drawingContext.setLineDash(list);
};

p5.prototype.plot = function(rect, y){
  this.beginShape();

  const w = rect[1][0] - rect[0][0];
  const h = rect[1][1] - rect[0][1];
  y = mth.div(y, mth.max(y));
  const x = mth.linspace(rect[0][0], rect[1][0], y.length);
  this.vertex(x[0], rect[1][1]);
  for (var i = 0; i < y.length; i++)
    this.vertex(x[i], rect[1][1] - y[i]*h);
  this.vertex(rect[1][0], rect[1][1]);
  this.endShape(this.CLOSE);
}

p5.prototype.center_image = function(img, x, y, w, h){
  this.image(img, x-w/2, y-h/2, w, h);
}

p5.prototype.show_fps = function(x=50, y=50) {
  this.text("fps:" + this.frameRate().toFixed(2), x, y);
}


class RawTextureWrapper extends p5.Texture {
  constructor(renderer, obj, w, h) {
    super(renderer, obj)
    this.width = w
    this.height = h
    return this
  }

  _getTextureDataFromSource() {
    console.log('get data source');
    console.log(this.src.width);
    return this.src;
  }

  init(tex) {
    const gl = this._renderer.GL
    this.glTex = tex

    this.glWrapS = this._renderer.textureWrapX
    this.glWrapT = this._renderer.textureWrapY

    this.setWrapMode(this.glWrapS, this.glWrapT)
    this.setInterpolation(this.glMinFilter, this.glMagFilter)
  }

  update() {
    return false;
  }
}

//
p5.prototype.create_fbo = function(canvas, name, w=0, h=0, depth=false) {
  // some code adapted from https://github.com/aferriss/p5Fbo/blob/master/p5Fbo.js
  // and https://editor.p5js.org/davepagurek/sketches/cmcqbj1II
  if (this.fbos == undefined){
    this.fbos = {};
  }

  let width = this.width || w;
  let height = this.width || w;
  width = width*canvas._pInst._pixelDensity;
  height = height*canvas._pInst._pixelDensity;

  let gl = canvas.GL;

  let im;
  if (name in this.fbos){
    im = this.fbos[name].im;
    im.resize(width, height);
    gl.deleteFrameBuffer(this.fbos[name].fbo);
  }else{
    im = this.createImage(width, height);
  }


  // const gltex = gl.createTexture();
  // gl.bindTexture(gl.TEXTURE_2D, gltex);
  // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  // gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);


  //const tex = new RawTextureWrapper(canvas, gltex, width, height);
  //const tex = new p5.Texture(canvas, im);
  this.texture(im);
  let tex = canvas.getTexture(im);;
  let fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex.glTex, 0);
  console.log(tex.glTex);

  //this.texture(null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  let fboobj = {
    tex:tex, im:im, fbo:fbo, canvas:canvas,
    bind: () => {
      //canvas._pInst.push();
      //canvas._tex = null;
      gl.bindTexture(gl.TEXTURE_2D, tex.glTex);

      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      //p5.texture(im);
      //p5.ortho(0, width, -height, 0);
      //gl.identity();
      gl.viewport(0, 0, width, height);
      //p5.ortho(0, w, -w, 0);
    },
    unbind: () => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      //canvas._pInst.pop();
      //console.log(canvas.width)
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    },

    draw: (x=0, y=0) => {
      this.texture(im);
      tex.bindTexture();
      this.noStroke();
      const size = [this.width, this.height];
      this.rect(0+x, size[1]+y, size[0], -size[1]);
    }

  };
  this.fbos[name] = fboobj;
  return fboobj;
}

module.exports = p5;
