"use strict"
const mth = require('./mth.js');
const color = function () {}

color.rgb_to_hsv = (rgba, range=255) => {
    rgba = mth.div(rgba, range);
    let [r, g, b] = rgba.slice(0,3);
    let a = 1;
    if (rgba.length > 3){
        a = rgba[3];
    }
    let K = 0;
    if (g < b){
        [g, b] = [b, g];
        K = -1;
    }

    if (r < g){
        [r, g] = [g, r];
        K = -2 / 6 - K;
    }
    const chroma = (g<b)?r - g:r - b;// #r - (g < b ? g : b);
    const h = Math.abs(K + (g - b) / (6 * chroma + 1e-20));
    const s = chroma / (r + 1e-20);
    const v = r;

    return [h,s,v,a].slice(0, rgba.length);
}

const fmod1 = (v) =>{
    if (v < 0)
        return v+1;
    if (v > 1)
        return v-1;
    return v;
}

color.hsv_to_rgb = (hsva, range=255) => {
    let [h, s, v] = hsva.slice(0,3);
    let a = 1;
    if (hsva.length > 3)
        a = hsva[3];
    let r,g,b;
    if (s == 0.0)
        r = g = b = v;
    else{
        h = fmod1(h) / (60.0 / 360.0);
        let i = Math.floor(h);
        let f = h - i;
        let p = v * (1.0 - s);
        let q = v * (1.0 - s * f);
        let t = v * (1.0 - s * (1.0 - f));

        switch(i){
            case 0:
                [r, g, b] = [v, t, p];
                break;
            case 1:
                [r, g, b] = [q, v, p];
                break;
            case 2:
                [r, g, b] = [p, v, t];
                break;
            case 3:
                [r, g, b] = [p, q, v];
                break;
            case 4:
                [r, g, b] = [t, p, v];
                break;
            default:
                [r, g, b] = [v, p, q];
        }
    }

    return mth.mul([r,g,b,a].slice(0, hsva.length), range);
}




color.categorical = {
    default12 : [[255, 109, 174],
  [212, 202, 58],
  [0, 190, 255],
  [235, 172, 250],
  [158, 158, 158],
  [103, 225, 181],
  [214, 39, 40],
  [154, 208, 255],
  [226, 133, 68],
  [31, 119, 180],
  [255, 127, 14],
  [93, 177, 90]]

};

color.current = color.categorical.default12;
color.default = (i) => color.current[i%color.current.length];
color.random = () => mth.random.choice(color.current);

const make_palette = (colors) => {
    return {
        color: (i) => palette[i%colors.length],
        length: colors.length,
        colors: colors
    };
}

// https://www.procjam.com/tutorials/en/color/
color.adic = (n, offset, s=1, v=1) => {
    let hues = [];
    let size = 1/n;
    for (let i=0; i < n; i++){
        let h = i*size;
        hues.push(np.array(fmod1(h+offset)));
    }
    return make_palette( hues.map(h=>color.hsv_to_rgb([h, s, v])) );
}

color.split_complementary = (h1, split=0.1, s=1, v=1, offset=0.5) => {
    let colors = [];
    let H = [h1,
         fmod1(fmod1(h1 + offset) + split),
             fmod1(fmod1(h1 + offset) - split)];
    return make_palette( H.map(h=>color.hsv_to_rgb([h, s, v])) );
}

color.rectangle = (h1, split=0.1, s=1, v=1) => {
    let colors = [];
    let H = [fmod1(h1 + split),
         fmod1(h1 - split),
         fmod1(fmod1(h1 + 0.5) + split),
         fmod1(fmod1(h1 - 0.5) - split)];
    return make_palette(H.map(h=>color.hsv_to_rgb([h, s, v])));
}

color.perturb = (rgb, h_offset=0.05, s_offset=0.1, v_offset=0.1) => {
    let [h, s, v] = color.rgb_to_hsv(rgb);
    h = fmod1(h + mth.random.normal()*h_offset);
    s = mth.clip(s - Math.abs(mth.random.normal()*s_offset), 0, 1);
    v = mth.clip(v - Math.abs(mth.random.normal()*v_offset), 0, 1);
    return color.hsv_to_rgb([h, s, v]);
}


color.randomize = (palette, n, h_offset=0.01, s_offset=0.2, v_offset=0.2) => {
    let res = [];
    for (let i=0; i < palette.length; i++){
        const rgb = palette.colors[i];//mth.random.choice(palette.colors);
        for (let j = 0; j < n; j++)
            res.push(color.perturb(rgb, h_offset, s_offset, v_offset));
    }
    res = mth.shuffled(res);
    return make_palette(res);
}

color.p5ImagePalette = (p5, path) => {
    let img = p5.loadImage(path);
    //img.loadPixels();
    let w = img.width;
    let h = img.height;

    return (x, y)=>{
        console.log(w);
        return img.get(Math.max(0, Math.min(Math.floor(x*(w-1)), w-1)),
                       Math.max(0, Math.min(Math.floor(y*(h-1)), h-1)));
    };
}

module.exports = color;
