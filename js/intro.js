import {fgsmTargeted, bimTargeted, jsmaOnePixel, jsma, cw} from './attacks.js';
import {MNIST_CLASSES, GTSRB_CLASSES, CIFAR_CLASSES, IMAGENET_CLASSES} from './class_names.js';

const $ = query => document.querySelector(query);

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
* Attach Event Handlers
************************************************************************/

// On page load
window.addEventListener('load', showImage);
window.addEventListener('load', resetAvailableAttacks);

// Model selection dropdown
$('#select-model').addEventListener("change", showImage);
$('#select-model').addEventListener("change", resetOnNewImage);
$('#select-model').addEventListener("change", resetAttack);

// Next image button
$('#next-image').addEventListener("click", showNextImage);
$('#next-image').addEventListener("click", resetOnNewImage);
$('#next-image').addEventListener("click", resetAttack);

// Predict button (original image)
$('#predict-original').addEventListener("click", predict);

// Target label dropdown
$('#select-target').addEventListener("change", resetAttack);

// Attack algorithm dropdown
$('#select-attack').addEventListener("change", resetAttack);

// Generate button
$('#generate-adv').addEventListener("click", generateAdv);

// Predict button (adversarial image)
$('#predict-adv').addEventListener("click", predictAdv);

/************************************************************************
* Define Event Handlers
************************************************************************/

/**
 * Renders the next image from the sample dataset in the original canvas
 */
function showNextImage() {
  let modelName = $('#select-model').value;
  if (modelName === 'mnist') { showNextMnist(); }
  else if (modelName === 'cifar') { showNextCifar(); }
  else if (modelName === 'gtsrb') { showNextGtsrb(); }
  else if (modelName === 'imagenet') { showNextImagenet(); }
}

/**
 * Renders the current image from the sample dataset in the original canvas
 */
function showImage() {
  let modelName = $('#select-model').value;
  if (modelName === 'mnist') { showMnist(); }
  else if (modelName === 'cifar') { showCifar(); }
  else if (modelName === 'gtsrb') { showGtsrb(); }
  else if (modelName === 'imagenet') { showImagenet(); }
}

/**
 * Computes & displays prediction of the current original image
 */
async function predict() {
  $('#predict-original').disabled = true;
  $('#predict-original').innerText = 'Loading...';

  let modelName = $('#select-model').value;
  if (modelName === 'mnist') {
    await loadMnistModel();
    await loadingMnist;
    let lblIdx = mnistDataset[mnistIdx].ys.argMax(1).dataSync()[0];
    _predict(mnistModel, mnistDataset[mnistIdx].xs, lblIdx, MNIST_CLASSES);
  } else if (modelName === 'cifar') {
    await loadCifarModel();
    await loadingCifar;
    let lblIdx = cifarDataset[cifarIdx].ys.argMax(1).dataSync()[0];
    _predict(cifarModel, cifarDataset[cifarIdx].xs, lblIdx, CIFAR_CLASSES);
  } else if (modelName === 'gtsrb') {
    await loadGtsrbModel();
    await loadingGtsrb;
    let lblIdx = gtsrbDataset[gtsrbIdx].ys.argMax(1).dataSync()[0];
    _predict(gtsrbModel, gtsrbDataset[gtsrbIdx].xs, lblIdx, GTSRB_CLASSES);
  } else if (modelName === 'imagenet') {
    await loadImagenetModel();
    await loadedImagenetData;
    _predict(imagenetModel, imagenetX[imagenetIdx], imagenetYLbls[imagenetIdx], IMAGENET_CLASSES);
  }

  $('#predict-original').innerText = 'Run Neural Network';

  function _predict(model, img, lblIdx, CLASS_NAMES) {
    // Generate prediction
    let pred = model.predict(img);
    let predLblIdx = pred.argMax(1).dataSync()[0];
    let predProb = pred.max().dataSync()[0];

    // Display prediction
    showPrediction(`Prediction: "${CLASS_NAMES[predLblIdx]}"<br/>Probability: ${predProb.toFixed(4)}`);
  }
 }

/**
 * Generates adversarial example from the current original image
 */
let advPrediction;
async function generateAdv() {
  $('#generate-adv').disabled = true;
  $('#generate-adv').innerText = 'Loading...';

  let attack;
  switch ($('#select-attack').value) {
    case 'fgsmTargeted': attack = fgsmTargeted; break;
    case 'bimTargeted': attack = bimTargeted; break;
    case 'jsmaOnePixel': attack = jsmaOnePixel; break;
    case 'jsma': attack = jsma; break;
    case 'cw': attack = cw; break;
  }

  let modelName = $('#select-model').value;
  let targetLblIdx = parseInt($('#select-target').value);
  if (modelName === 'mnist') {
    await loadMnistModel();
    await loadingMnist;
    await _generateAdv(mnistModel, mnistDataset[mnistIdx].xs, mnistDataset[mnistIdx].ys, MNIST_CLASSES);
  } else if (modelName === 'cifar') {
    await loadCifarModel();
    await loadingCifar;
    await _generateAdv(cifarModel, cifarDataset[cifarIdx].xs, cifarDataset[cifarIdx].ys, CIFAR_CLASSES);
  } else if (modelName === 'gtsrb') {
    await loadGtsrbModel();
    await loadingGtsrb;
    await _generateAdv(gtsrbModel, gtsrbDataset[gtsrbIdx].xs, gtsrbDataset[gtsrbIdx].ys, GTSRB_CLASSES);
  } else if (modelName === 'imagenet') {
    await loadImagenetModel();
    await loadedImagenetData;
    await _generateAdv(imagenetModel, imagenetX[imagenetIdx], imagenetY[imagenetIdx], IMAGENET_CLASSES);
  }

  $('#generate-adv').innerText = 'Generate';

  async function _generateAdv(model, img, lbl, CLASS_NAMES) {
    // Generate adversarial example
    let targetLbl = tf.oneHot(targetLblIdx, lbl.shape[1]).reshape(lbl.shape);
    let aimg = tf.tidy(() => attack(model, img, lbl, targetLbl));

    // Display adversarial example
    $('#difference').style.display = 'block';
    await drawImg(aimg, 'adversarial');

    // Compute & store adversarial prediction
    let pred = model.predict(aimg);
    let predLblIdx = pred.argMax(1).dataSync()[0];
    let predProb = pred.max().dataSync()[0];
    let lblIdx = lbl.argMax(1).dataSync()[0];
    let msg;
    if (predLblIdx === targetLblIdx) {
      msg = 'Attack succeeded'
    } else if (predLblIdx !== lblIdx) {
      msg = 'Attack partially succeeded'
    } else {
      msg = 'Attack failed'
    }
    advPrediction = `${msg}<br/>Prediction: "${CLASS_NAMES[predLblIdx]}"<br/>Probability: ${predProb.toFixed(4)}`;
  }
}

/**
 * Displays prediction of the current adversarial image
 * (This function just renders the status we've already computed in generateAdv())
 */
function predictAdv() {
  $('#predict-adv').disabled = true;
  showAdvPrediction(advPrediction);
}

/**
 * Reset entire dashboard UI when a new image is selected
 */
function resetOnNewImage() {
  $('#predict-original').disabled = false;
  $('#predict-original').innerText = 'Run Neural Network';
  $('#prediction').innerHTML = '';
  resetAttack();
  resetAvailableAttacks();
}

/**
 * Reset attack UI when a new target label, attack, or image is selected
 */
function resetAttack() {
  $('#generate-adv').disabled = false;
  $('#predict-adv').disabled = false;
  $('#difference').style.display = 'none';
  $('#prediction-adv').innerHTML = '';
  drawImg(tf.ones([1, 224, 224, 1]), 'adversarial');
}

/**
 * Reset available attacks and target labels when a new image is selected
 */
function resetAvailableAttacks() {
  const MNIST_TARGETS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const CIFAR_TARGETS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const GTSRB_TARGETS = [8, 0, 14, 17];
  const IMAGENET_TARGETS = [934, 413, 151];

  let modelName = $('#select-model').value;
  if (modelName === 'mnist') {
    let originalLbl = mnistDataset[mnistIdx].ys.argMax(1).dataSync()[0];
    _resetAvailableAttacks(true, originalLbl, MNIST_TARGETS, MNIST_CLASSES);
  } else if (modelName === 'cifar') {
    let originalLbl = cifarDataset[cifarIdx].ys.argMax(1).dataSync()[0];
    _resetAvailableAttacks(true, originalLbl, CIFAR_TARGETS, CIFAR_CLASSES);
   }
  else if (modelName === 'gtsrb') {
    let originalLbl = gtsrbDataset[gtsrbIdx].ys.argMax(1).dataSync()[0];
    _resetAvailableAttacks(false, originalLbl, GTSRB_TARGETS, GTSRB_CLASSES);
  }
  else if (modelName === 'imagenet') {
    let originalLbl = imagenetYLbls[imagenetIdx];
    _resetAvailableAttacks(false, originalLbl, IMAGENET_TARGETS, IMAGENET_CLASSES);
  }

  function _resetAvailableAttacks(jsma, originalLbl, TARGETS, CLASS_NAMES) {
    let modelName = $('#select-model').value;
    let selectAttack = $('#select-attack');
    let selectTarget = $('#select-target');

    // Add or remove JSMA as an option
    if (jsma === true) {
      selectAttack.querySelector('option[value=jsma]').disabled = false;
    } else {
      selectAttack.querySelector('option[value=jsma]').disabled = true;
      if (selectAttack.value === 'jsma') { selectAttack.value = 'fgsmTargeted'; }
    }

    // Filter available target classes in dropdown
    if (selectTarget.getAttribute('data-model') === modelName) {
      // Go through options and disable the current class as a target class
      selectTarget.options.forEach(option => {
        if (parseInt(option.value) === originalLbl) { option.disabled = true; }
        else {option.disabled = false; }
      });
      // Reset the selected option if it's now disabled
      if (parseInt(selectTarget.value) === originalLbl) {
        selectTarget.options[0].selected = true;
        if (parseInt(selectTarget.value) === originalLbl) {
          selectTarget.options[1].selected = true;
        }
      }
    } else {
      // Rebuild options from scratch (b/c the user chose a new model)
      selectTarget.innerHTML = '';
      TARGETS.forEach(i => {
        let option = new Option(CLASS_NAMES[i], i);
        if (i === originalLbl) { option.disabled = true; }
        selectTarget.appendChild(option);
      });
      selectTarget.setAttribute('data-model', modelName);
    }
  }
}

/************************************************************************
* Visualize Attacks
************************************************************************/

function showPrediction(msg) {
  $('#prediction').innerHTML = msg;
}

function showAdvPrediction(msg) {
  $('#prediction-adv').innerHTML = msg;
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
