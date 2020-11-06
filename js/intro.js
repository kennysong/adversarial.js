import {fgsmTargeted, bimTargeted, jsmaOnePixel, jsma, cw} from './attacks.js';
import {GTSRB_CLASSES, CIFAR_CLASSES, IMAGENET_CLASSES} from './class_names.js';

/************************************************************************
* Load Datasets
************************************************************************/

/****************************** Load MNIST ******************************/

let mnistDataset;
let mnistUrl = 'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/mnist_ten.csv?alt=media';
let loadingMnist = tf.data.csv(mnistUrl, {columnConfigs: {label: {isLabel: true}}})
  .map(({xs, ys}) => {
    xs = Object.values(xs).map(e => e/255);  // Convert from feature object to array, and normalize
    ys = tf.oneHot(Object.values(ys), 10).squeeze();  // Convert from feature object to scalar, and turn into one-hot vector
    return {xs: xs, ys: ys};
  })
  .batch(1)
  .toArray()
  .then(ds => mnistDataset = ds);

/****************************** Load CIFAR-10 ******************************/

let cifarXUrl = 'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/cifar10_sample_x_1.json?alt=media';
let cifarYUrl = 'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/cifar10_sample_y_1.json?alt=media';

// Load data in form [{xs: x0_tensor, ys: y0_tensor}, {xs: x1_tensor, ys: y1_tensor}, ...]
let cifarX, cifarY, cifarDataset;
let loadingCifarX = fetch(cifarXUrl).then(res => res.json()).then(arr => cifarX = tf.data.array(arr).batch(1));
let loadingCifarY = fetch(cifarYUrl).then(res => res.json()).then(arr => cifarY = tf.data.array(arr).batch(1));
let loadingCifar = Promise.all([loadingCifarX, loadingCifarY]).then(() => tf.data.zip([cifarX, cifarY]).toArray()).then(ds => cifarDataset = ds.map(e => { return {xs: e[0], ys: e[1]}}));

/****************************** Load GTSRB ******************************/

let gtsrbXUrl = 'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/gtsrb_sample_x_4.json?alt=media';
let gtsrbYUrl = 'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/gtsrb_sample_y_4.json?alt=media';

// Load data in form [{xs: x0_tensor, ys: y0_tensor}, {xs: x1_tensor, ys: y1_tensor}, ...]
let gtsrbX, gtsrbY, gtsrbDataset;
let loadingGtsrbX = fetch(gtsrbXUrl).then(res => res.json()).then(arr => gtsrbX = tf.data.array(arr).batch(1));
let loadingGtsrbY = fetch(gtsrbYUrl).then(res => res.json()).then(arr => gtsrbY = tf.data.array(arr).batch(1));
let loadingGtsrb = Promise.all([loadingGtsrbX, loadingGtsrbY]).then(() => tf.data.zip([gtsrbX, gtsrbY]).toArray()).then(ds => gtsrbDataset = ds.map(e => { return {xs: e[0], ys: e[1]}}));

/****************************** Load ImageNet ******************************/

let imagenetXUrls = [
  'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/574_golf_ball.jpg?alt=media',
  'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/217_english_springer.jpg?alt=media',
  'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/701_parachute.jpg?alt=media',
  'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/0_tench.jpg?alt=media',
  'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/497_church.jpg?alt=media',
  'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/566_french_horn.jpg?alt=media'
]
let imagenetYLbls = [574, 217, 701, 0, 497, 566]
let imagenetY = imagenetYLbls.map(lbl => tf.oneHot(lbl, 1000).reshape([1, 1000]));

// Utility function that loads an image in a given <img> tag and returns a Promise
function loadImage(e, url) {
  return new Promise((resolve) => {
    e.addEventListener('load', () => resolve(e));
    e.src = url;
  });
}

// Load each image
let loadingImagenetX = [];
for (let i = 0; i < imagenetXUrls.length; i++) {
  document.getElementsByClassName(i.toString()).forEach(e => {
    let loadingImage = loadImage(e, imagenetXUrls[i]);
    loadingImagenetX.push(loadingImage);
  });
}

// Collect pixel data from each image
let imagenetX = [];
let loadedImagenetData = Promise.all(loadingImagenetX);
loadedImagenetData.then(() => {
  for (let i = 0; i < imagenetXUrls.length; i++) {
    let img = document.getElementsByClassName(i.toString())[0];
    imagenetX.push(tf.browser.fromPixels(img).div(255.0).reshape([1, 224, 224, 3]));
  }
});

/************************************************************************
* Load Models
************************************************************************/

/****************************** Load MNIST ******************************/

let mnistModel;
async function loadMnistModel() {
  if (mnistModel !== undefined) { return; }
  mnistModel = await tf.loadLayersModel('https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/mnist_dnn.json?alt=media');
}

/****************************** Load CIFAR-10 ******************************/

let cifarModel;
async function loadCifarModel() {
  if (cifarModel !== undefined) { return; }
  cifarModel = await tf.loadLayersModel('https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/cifar10_cnn_2.json?alt=media');
}

/****************************** Load GTSRB ******************************/

let gtsrbModel;
async function loadGtsrbModel() {
  if (gtsrbModel !== undefined) { return; }
  gtsrbModel = await tf.loadLayersModel('https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/gtsrb_cnn_3.json?alt=media');
}

/****************************** Load ImageNet ******************************/

let imagenetModel;
async function loadImagenetModel() {
  if (imagenetModel !== undefined) { return; }
  imagenetModel = await mobilenet.load({version: 2, alpha: 1.0});

  // Monkey patch the mobilenet object to have a predict() method like a normal tf.LayersModel
  imagenetModel.predict = function (img) {
    return this.predictLogits(img).softmax();
  }

  // Also monkey patch the mobilenet object with a method to predict logits
  imagenetModel.predictLogits = function (img) {
    // Remove the first "background noise" logit
    // Copied from: https://github.com/tensorflow/tfjs-models/blob/708e3911fb01d0dfe70448acc3e8ca736fae82d3/mobilenet/src/index.ts#L232
    const logits1001 = this.model.predict(img);
    return logits1001.slice([0, 1], [-1, 1000]);
  }
}

/************************************************************************
* Event Handlers
************************************************************************/

window.addEventListener('load', showImage);

document.getElementById('select-model').addEventListener("change", showImage);
document.getElementById('select-model').addEventListener("change", resetOnNewImage);
document.getElementById('select-model').addEventListener("change", resetAttack);

document.getElementById('next-image').addEventListener("click", showNextImage);
document.getElementById('next-image').addEventListener("click", resetOnNewImage);
document.getElementById('next-image').addEventListener("click", resetAttack);

document.getElementById('predict-original').addEventListener("click", predict);

document.getElementById('select-label-mnist').addEventListener("change", resetAttack);

document.getElementById('generate-adv').addEventListener("click", attack);

document.getElementById('predict-adv').addEventListener("click", predictAdv);

function showNextImage() {
  let modelName = document.getElementById('select-model').value;
  if (modelName === 'mnist') { showNextMnist(); }
  else if (modelName === 'cifar') { showNextCifar(); }
  else if (modelName === 'gtsrb') { showNextGtsrb(); }
  else if (modelName === 'imagenet') { showNextImagenet(); }
}

function showImage() {
  let modelName = document.getElementById('select-model').value;
  if (modelName === 'mnist') { showMnist(); }
  else if (modelName === 'cifar') { showCifar(); }
  else if (modelName === 'gtsrb') { showGtsrb(); }
  else if (modelName === 'imagenet') { showImagenet(); }
}

async function predict() {
  document.getElementById('predict-original').disabled = true;
  document.getElementById('predict-original').innerText = 'Loading...';

  let modelName = document.getElementById('select-model').value;
  if (modelName === 'mnist') { await _predictMnist(); }
  else if (modelName === 'cifar') { await _predictCifar(); }
  else if (modelName === 'gtsrb') { await _predictGtsrb(); }
  else if (modelName === 'imagenet') { await _predictImagenet(); }

  document.getElementById('predict-original').innerText = 'Predict';

  async function _predictMnist() {
    // Load model & data
    await loadMnistModel();
    await loadingMnist;

    // Generate prediction
    let img = mnistDataset[mnistIdx].xs;
    let pred = mnistModel.predict(img);
    let predLblIdx = pred.argMax(1).dataSync()[0];
    let predProb = pred.max().dataSync()[0];

    // Display prediction
    let trueClass = mnistDataset[mnistIdx].ys.argMax(1).dataSync()[0];
    showPrediction(`Prediction: "${predLblIdx}"<br/>Probability: ${predProb.toFixed(4)}`);
  }

  async function _predictCifar() {
    // Load model & data
    await loadCifarModel();
    await loadingCifar;

    // Generate prediction
    let img = cifarDataset[cifarIdx].xs;
    let pred = cifarModel.predict(img);
    let predLblIdx = pred.argMax(1).dataSync()[0];
    let predProb = pred.max().dataSync()[0];

    // Display prediction
    let trueClass = cifarDataset[cifarIdx].ys.argMax(1).dataSync()[0];
    showPrediction(`Prediction: "${CIFAR_CLASSES[predLblIdx]}"<br/>Probability: ${predProb.toFixed(4)}`);
  }

  async function _predictGtsrb() {
    // Load model & data
    await loadGtsrbModel();
    await loadingGtsrb;

    // Generate prediction
    let img = gtsrbDataset[gtsrbIdx].xs;
    let pred = gtsrbModel.predict(img);
    let predLblIdx = pred.argMax(1).dataSync()[0];
    let predProb = pred.max().dataSync()[0];

    // Display prediction
    let trueClass = gtsrbDataset[gtsrbIdx].ys.argMax(1).dataSync()[0];
    showPrediction(`Prediction: "${GTSRB_CLASSES[predLblIdx]}"<br/>Probability: ${predProb.toFixed(4)}`);
  }

  async function _predictImagenet() {
    // Load model & data
    await loadImagenetModel();
    await loadedImagenetData;

    // Generate prediction
    let img = imagenetX[imagenetIdx];
    let pred = imagenetModel.predict(img);
    let predLblIdx = pred.argMax(1).dataSync()[0];
    let predProb = pred.max().dataSync()[0];

    // Display prediction
    let trueClass = imagenetYLbls[imagenetIdx];
    showPrediction(`Prediction: "${IMAGENET_CLASSES[predLblIdx]}"<br/>Probability: ${predProb.toFixed(4)}`);
  }
}

let advPrediction;
async function attack() {
  document.getElementById('generate-adv').disabled = true;
  document.getElementById('generate-adv').innerText = 'Loading...';

  let modelName = document.getElementById('select-model').value;
  if (modelName === 'mnist') {
    // Load model & data
    await loadMnistModel();
    await loadingMnist;
    let model = mnistModel;
    let img = mnistDataset[mnistIdx].xs;
    let lbl = mnistDataset[mnistIdx].ys;
    let lblIdx = lbl.argMax(1).dataSync()[0];

    // Generate adversarial example
    let targetLblIdx = parseInt(document.getElementById(`select-label-${modelName}`).value);
    let targetLbl = tf.oneHot(targetLblIdx, 10).reshape([1, 10]);
    let aimg = tf.tidy(() => fgsmTargeted(model, img, lbl, targetLbl));

    // Display adversarial example
    document.getElementById('difference').style.display = 'block';
    await drawImg(aimg, 'adversarial');

    // Display adversarial prediction
    let pred = model.predict(aimg);
    let predLblIdx = pred.argMax(1).dataSync()[0];
    let predProb = pred.max().dataSync()[0];
    let msg;
    if (predLblIdx === targetLblIdx) {
      msg = 'Attack succeeded'
    } else if (predLblIdx !== lblIdx) {
      msg = 'Attack partially succeeded'
    } else {
      msg = 'Attack failed'
    }
    advPrediction = `${msg}<br/>Prediction: "${predLblIdx}"<br/>Probability: ${predProb.toFixed(4)}`;
  } else if (modelName === 'cifar') {
    // Load model & data
    await loadCifarModel();
    await loadingCifar;
  } else if (modelName === 'gtsrb') {
    // Load model & data
    await loadGtsrbModel();
    await loadingGtsrb;
  } else if (modelName === 'imagenet') {
    // Load model & data
    await loadImagenetModel();
    await loadedImagenetData;
  }

  document.getElementById('generate-adv').innerText = 'Attack';
}

function predictAdv() {
  // This function just renders the status we've already computed and stored
  document.getElementById('predict-adv').disabled = true;
  showAdvPrediction(advPrediction);
}

function resetOnNewImage() {
  document.getElementById('predict-original').disabled = false;
  document.getElementById('predict-original').innerText = 'Predict';
  document.getElementById('prediction').innerHTML = '';
  resetAttack();
}

function resetAttack() {
  document.getElementById('generate-adv').disabled = false;
  document.getElementById('predict-adv').disabled = false;
  document.getElementById('difference').style.display = 'none';
  document.getElementById('prediction-adv').innerHTML = '';
  drawImg(tf.ones([1, 224, 224, 1]), 'adversarial');
}

/************************************************************************
* Visualize Attacks
************************************************************************/

function showPrediction(msg) {
  document.getElementById('prediction').innerHTML = msg;
}

function showAdvPrediction(msg) {
  document.getElementById('prediction-adv').innerHTML = msg;
}

let mnistIdx = 0;
async function showMnist() {
  await loadingMnist;
  await drawImg(mnistDataset[mnistIdx].xs, 'original');
}
async function showNextMnist() {
  mnistIdx = (mnistIdx + 1) % mnistDataset.length;
  await showMnist();
}

let cifarIdx = 0;
async function showCifar() {
  await loadingCifar;
  await drawImg(cifarDataset[cifarIdx].xs, 'original');
}
async function showNextCifar() {
  cifarIdx = (cifarIdx + 1) % cifarDataset.length;
  await showCifar();
}

let gtsrbIdx = 0;
async function showGtsrb() {
  await loadingGtsrb;
  await drawImg(gtsrbDataset[gtsrbIdx].xs, 'original');
}
async function showNextGtsrb() {
  gtsrbIdx = (gtsrbIdx + 1) % gtsrbDataset.length;
  await showGtsrb();
}

let imagenetIdx = 0;
async function showImagenet() {
  await loadingImagenetX;
  await drawImg(imagenetX[imagenetIdx], 'original');
}
async function showNextImagenet() {
  imagenetIdx = (imagenetIdx + 1) % imagenetX.length;
  await showImagenet();
}

async function drawImg(img, element, msg, border) {
  // Draw image
  let canvas = document.getElementById(element);
  if (img.shape[1] === 784) {
    let resizedImg = tf.image.resizeNearestNeighbor(img.reshape([28, 28, 1]), [224, 224]);
    await tf.browser.toPixels(resizedImg, canvas);
  } else {
    let resizedImg = tf.image.resizeNearestNeighbor(img.squeeze(0), [224, 224]);
    await tf.browser.toPixels(resizedImg, canvas);
  }

  // Draw caption
  if (msg !== undefined) {
    canvas.nextSibling.innerHTML = msg;
  } else {
    canvas.nextSibling.innerHTML = '<br/>';
  }

  // Draw border
  if (border !== undefined) {
    canvas.style.borderColor = border;
    canvas.style.borderWidth = '5px';
  } else {
    canvas.style.borderColor = 'black';
    canvas.style.borderWidth = '1px';
  }
}
