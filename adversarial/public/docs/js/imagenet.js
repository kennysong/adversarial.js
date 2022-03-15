import {IMAGENET_CLASSES} from './class_names.js';

/************************************************************************
* Load Model
************************************************************************/

// Note: The asynchronous calls are a little bit funky. Data is loaded immediately
// at page load, since it's easy and fast. The `loadingX` promise tracks that the
// data loaded successfully. The model is a bit too heavy to load immediately, since
// it'll block the page for a few seconds. Instead, we manually call loadModel()
// when the user clicks the button. There is no promise that tracks the status of
// the model load, since we just await on loadModel() in the inline script in
// imagenet.html.

let model;
export async function loadModel() {
  if (model !== undefined) { return; }
  model = await mobilenet.load({version: 2, alpha: 1.0});

  // Monkey patch the mobilenet object to have a predict() method like a normal tf.LayersModel
  model.predict = function (img) {
    return this.predictLogits(img).softmax();
  }

  // Also monkey patch the mobilenet object with a method to predict logits
  model.predictLogits = function (img) {
    // Remove the first "background noise" logit
    // Copied from: https://github.com/tensorflow/tfjs-models/blob/708e3911fb01d0dfe70448acc3e8ca736fae82d3/mobilenet/src/index.ts#L232
    const logits1001 = this.model.predict(img);
    return logits1001.slice([0, 1], [-1, 1000]);
  }
}

/************************************************************************
 * Load Dataset
 ************************************************************************/

let xUrls = [
  'data/imagenet/574_golf_ball.jpg',
  'data/imagenet/217_english_springer.jpg',
  'data/imagenet/701_parachute.jpg',
  'data/imagenet/0_tench.jpg',
  'data/imagenet/497_church.jpg',
  'data/imagenet/566_french_horn.jpg'
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
let loadedData = Promise.all(loadingX);
loadedData.then(() => {
  for (let i = 0; i < xUrls.length; i++) {
    let img = document.getElementsByClassName(i.toString())[0];
    x.push(tf.browser.fromPixels(img).div(255.0).reshape([1, 224, 224, 3]));
  }
});

/************************************************************************
 * Visualize Attacks
 ************************************************************************/

const CONFIGS = {
  'fgsm': {ε: 0.05},  // 0.1 L_inf perturbation is too visible in color
  'fgsmTargeted': {loss: 1},  // The 2nd loss function is too heavy for ImageNet
  'jsmaOnePixel': {ε: 75},  // This is unsuccessful. I estimate that it requires ~50x higher ε than CIFAR-10 to be successful on ImageNet, but that is too slow
  'cw': {κ: 5, c: 1, λ: 0.05}  // Generate higher confidence adversarial examples, and minimize distortion
};

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
  await loadedData;
  let successes = 0;
  let NUM_ROWS = x.length;

  for (let i = 0; i < NUM_ROWS; i++) {  // For each row
    let img = x[i];
    let lbl = y[i];
    let lblIdx = yLbls[i];

    // Compute and display original class probability
    let p = model.predict(img).dataSync()[lblIdx];
    let status = document.getElementById(attack.name).getElementsByClassName(i.toString())[0].nextSibling;
    status.innerHTML = `Prediction: ${IMAGENET_CLASSES[lblIdx]}<br/>Prob: ${p.toFixed(3)}`;

    // Generate adversarial image from attack
    let aimg = tf.tidy(() => attack(model, img, lbl, CONFIGS[attack.name]));

    // Display adversarial image and its probability
    p = model.predict(aimg).max(1).dataSync()[0];
    let albl = model.predict(aimg).argMax(1).dataSync()[0];
    if (albl !== lblIdx) {
      successes++;
      await drawImg(aimg, `${i}a`, attack.name, `Prediction: ${IMAGENET_CLASSES[albl]}<br/>Prob: ${p.toFixed(3)}`, true);
    } else {
      await drawImg(aimg, `${i}a`, attack.name, `Prediction: ${IMAGENET_CLASSES[albl]}<br/>Prob: ${p.toFixed(3)}`);
    }
  }

  document.getElementById(`${attack.name}-success-rate`).innerText = `Success rate: ${(successes / NUM_ROWS).toFixed(1)}`;
}

export async function runTargeted(attack) {
  await loadedData;
  let successes = 0;
  let targetLblIdxs = [934, 413, 151];

  let NUM_ROWS = x.length;
  let NUM_COLS = targetLblIdxs.length;

  for (let i = 0; i < NUM_ROWS; i++) {  // For each row
    let img = x[i];
    let lbl = y[i];
    let lblIdx = yLbls[i];

    // Compute and display original class probability
    let p = model.predict(img).dataSync()[lblIdx];
    let status = document.getElementById(attack.name).getElementsByClassName(i.toString())[0].nextSibling;
    status.innerHTML = `Prediction: ${IMAGENET_CLASSES[lblIdx]}<br/>Prob: ${p.toFixed(3)}`;

    for (let j = 0; j < NUM_COLS; j++) {  // For each target label
      let targetLblIdx = targetLblIdxs[j];
      let targetLbl = tf.oneHot(targetLblIdx, 1000).reshape([1, 1000]);

      // Generate adversarial image from attack
      let aimg = tf.tidy(() => attack(model, img, lbl, targetLbl, CONFIGS[attack.name]));

      // Display adversarial image and its probability
      p = model.predict(aimg).dataSync()[targetLblIdx];
      let predLbl = model.predict(aimg).argMax(1).dataSync()[0];
      if (predLbl === targetLblIdx) {
        successes++;
        await drawImg(aimg, `${i}${j}`, attack.name, `Prediction: ${IMAGENET_CLASSES[targetLblIdx]}<br/>Prob: ${p.toFixed(3)}`, true);
      } else {
        await drawImg(aimg, `${i}${j}`, attack.name, `Prediction: ${IMAGENET_CLASSES[targetLblIdx]}<br/>Prob: ${p.toFixed(3)}`);
      }
    }
  }

  document.getElementById(`${attack.name}-success-rate`).innerText = `Success rate: ${(successes / (NUM_ROWS*NUM_COLS)).toFixed(2)}`;
}

