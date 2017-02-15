Neural Style
===================

A simple implementaton of the style transfer algorithm described here: https://arxiv.org/abs/1508.06576

----------
###Samples (512x512):
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/jf1.png" height="300px">

---
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/original-images/natalie_dormer.jpeg" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/original-images/fmi.jpg" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/original-images/chicago.jpg" height="210px">

<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/nw.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/f.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/cw.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/original-images/wave_crop.jpg" height="210px">

<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/nc.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/fc.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/cc.png" height="210x">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/original-images/candy.jpg" height="210px">

<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/nst.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/fs.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/cs1.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/original-images/starry_night_crop.jpg" width="240px">

<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/nf.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/ff.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/cf.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/original-images/fire.jpg" height="210px">

---
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/jf2.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/fpb.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/ft.png" height="210px">

<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/es.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/cs1.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/cs.png" height="210px">
<img src="https://raw.githubusercontent.com/DanailKoychev/neural-style/master/sample-images/ns.png" height="210px">

---

###Requirements
pyhon 3.5

tensorflow (preferably with GPU support)

tensorflow-vgg (https://github.com/machrisaa/tensorflow-vgg)

numpy, pyplot, sk-image

###Use
> python style_transfer.py "path/to/content.jpg" "path/to/style.jpg" 

options:
> --save_path "path/to/save/output"

> --iterations 200      # how long to run the optimizer

> --resolution 512     # pixels (square image)

> --content_weight 1000 # tunes tradeoff between content and style

> --style_weight 1 # tunes tradeoff between content and style

> --verbose # prints intermediate loss function values

The optimal content and style weight values may be different for every image combination. Some trial and error may be needed.
Too large content weight may cause the optimisation to stall from the beginning.

###Minor issues
Generating high resolution images may fail on low-memory GPUs.

Currently output images are drawn with matplotlib and include small x and y axes on the side (I'll fix it soon).

---
COMING SOON:
---------------------
Fast neural style 

described here: http://cs.stanford.edu/people/jcjohns/eccv16/, and here: https://arxiv.org/pdf/1607.08022.pdf
