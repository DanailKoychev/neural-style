Neural Style
===================

A simple implementaton of the style transfer algorithm described here: https://arxiv.org/abs/1508.06576

---

  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/jf1.png" height="300px">

---

<p float="left">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/original-images/natalie_dormer.jpeg" height="210px">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/original-images/fmi.jpg" height="210px">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/original-images/chicago.jpg" height="210px">
</p>

---

<p float="left">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/nw.png" height="210px">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/f.png" height="210px">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/cw.png" height="210px">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/original-images/wave_crop.jpg" height="210px">
</p>

---

<p float="left">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/nc.png" height="210px">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/fc.png" height="210px">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/cc.png" height="210x">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/original-images/candy.jpg" height="210px">
</p>

---

<p float="left">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/nst.png" height="210px">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/fs.png" height="210px">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/cs.png" height="210px">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/original-images/starry_night_crop.jpg" width="210px">
</p>

---

<p float="left">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/nf.png" height="210px">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/ff.png" height="210px">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/cf.png" height="210px">
  <img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/original-images/fire.jpg" height="210px">
</p>

---

Requires tensorflow-vgg (https://github.com/machrisaa/tensorflow-vgg) including pretrained weights

### Use
> python style_transfer.py "path/to/content.jpg" "path/to/style.jpg" 

options:
> --save_path &nbsp;&nbsp;# "path/to/save/output"

> --iterations 200 &nbsp;&nbsp;# how long to run the optimizer

> --resolution 512 &nbsp;&nbsp;# pixels (square image)

> --content_weight 1000 &nbsp;&nbsp;# tunes tradeoff between content and style

> --style_weight 1 &nbsp;&nbsp;# tunes tradeoff between content and style

The optimal content and style weight values may be different for every image combination.

