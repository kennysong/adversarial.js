import {IMAGENET_CLASSES} from './imagenet_classes.js';

/************************************************************************
* Load Model
************************************************************************/

let model;
let loadingModel = mobilenet.load({version: 2, alpha: 1.0}).then(m => {
  model = m;

  // Monkey patch the mobilenet object to have a predict() method like a normal tf.LayersModel
  model.predict = function (img) {
    // Remove the first "background noise" logit
    // Copied from: https://github.com/tensorflow/tfjs-models/blob/708e3911fb01d0dfe70448acc3e8ca736fae82d3/mobilenet/src/index.ts#L232
    const logits1001 = this.model.predict(img);
    return logits1001.slice([0, 1], [-1, 1000]).softmax();
  }
});

/************************************************************************
 * Load Dataset
 ************************************************************************/

let xUrls = [
  'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/574_golf_ball.jpg?alt=media',
  'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/217_english_springer.jpg?alt=media',
  'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/701_parachute.jpg?alt=media',
  'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/0_tench.jpg?alt=media',
  'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/497_church.jpg?alt=media',
  'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/566_french_horn.jpg?alt=media'
]
let yLbls = [574, 217, 701, 0, 497, 566]
let y = yLbls.map(lbl => tf.oneHot(lbl, 1000).reshape([1, 1000]));

// Utility function that loads an image in a given <img> tag and returns a Promise
function loadImage(e, url) {
  return new Promise((resolve) => {
    e.addEventListener('load', () => resolve(e));
    e.src = url;
  });
}

// Load each image
let loadingX = [];
for (let i = 0; i < xUrls.length; i++) {
  document.getElementsByClassName(i.toString()).forEach(e => {
    let loadingImage = loadImage(e, xUrls[i]);
    loadingX.push(loadingImage);
  });
}

// Collect pixel data from each image
let x = [];
Promise.all(loadingX).then(() => {
  for (let i = 0; i < xUrls.length; i++) {
    let img = document.getElementsByClassName(i.toString())[0];
    x.push(tf.browser.fromPixels(img).div(255.0).reshape([1, 224, 224, 3]));
  }
});

// Promise that resolves after data and model are both loaded
let allLoaded = Promise.all(loadingX.concat(loadingModel));

/************************************************************************
 * Visualize Attacks
 ************************************************************************/

async function drawImg(img, element, attackName, msg, success) {
  let canvas = document.getElementById(attackName).getElementsByClassName(element)[0];
  await tf.browser.toPixels(img.reshape([224, 224, 3]), canvas);

  if (msg !== undefined) {
    canvas.nextSibling.innerHTML = msg;
  }
  if (success === true) {
    canvas.style.borderColor = 'lime';
    canvas.style.borderWidth = '2px';
  }
}

export async function runUntargeted(attack) {
  document.body.offsetLeft;
  await allLoaded;
  let successes = 0;

  for (let i = 0; i < 6; i++) {  // For each row
    let img = x[i];
    let lbl = y[i];
    let lblIdx = yLbls[i];

    // Compute and display original class probability
    let p = model.predict(img).dataSync()[lblIdx];
    let status = document.getElementById(attack.name).getElementsByClassName(i.toString())[0].nextSibling;
    status.innerHTML = `Class: ${IMAGENET_CLASSES[lblIdx]}<br/>Prob: ${p.toFixed(3)}`;

    // Generate adversarial image from attack
    let aimg = tf.tidy(() => attack(model, img, lbl));

    // Display adversarial image and its probability
    p = model.predict(aimg).max(1).dataSync()[0];
    let albl = model.predict(aimg).argMax(1).dataSync()[0];
    if (albl !== lblIdx) {
      successes++;
      await drawImg(aimg, `${i}a`, attack.name, `Class: ${IMAGENET_CLASSES[albl]}<br/>Prob: ${p.toFixed(3)}`, true);
    }
    await drawImg(aimg, `${i}a`, attack.name, `Class: ${IMAGENET_CLASSES[albl]}<br/>Prob: ${p.toFixed(3)}`);
  }

  document.getElementById(`${attack.name}-success-rate`).innerText = `Success rate: ${(successes / 6).toFixed(1)}`;
}
