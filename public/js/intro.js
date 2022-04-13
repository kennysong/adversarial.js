import {fgsmTargeted, bimTargeted, jsmaOnePixel, jsma, cw} from './adversarial.js';
import {MNIST_CLASSES, GTSRB_CLASSES, CIFAR_CLASSES, IMAGENET_CLASSES} from './class_names.js';

/* eslint-disable no-unused-vars */
import * as tf from '../../node_modules/@tensorflow/tfjs';
import * as mobilenet from '../../node_modules/@tensorflow-models/mobilenet';
/************************************************************************
* Global constants
************************************************************************/

const $ = query => document.querySelector(query);

const MNIST_CONFIGS = {
  'fgsmTargeted': {ε: 0.2},  // Targeted FGSM works slightly better on MNIST with higher distortion
  'bimTargeted': {iters: 20},  // Targeted BIM works slightly better on MNIST with more iterations (pushes misclassification confidence up)
};

const GTSRB_CONFIGS = {
  'bimTargeted': {iters: 50},  // Needs more iterations to work well
  'jsmaOnePixel': {ε: 75},  // Works well with the same settings as CIFAR-10
};

const CIFAR_CONFIGS = {
  'fgsm': {ε: 0.05},  // 0.1 L_inf perturbation is too visible in color
  'jsmaOnePixel': {ε: 75},  // JSMA one-pixel on CIFAR-10 requires more ~3x pixels than MNIST
  'jsma': {ε: 75},  // JSMA on CIFAR-10 also requires more ~3x pixels than MNIST
  'cw': {c: 1, λ: 0.05}  // Tried to minimize distortion, but not sure it worked
};

const IMAGENET_CONFIGS = {
  'fgsm': {ε: 0.05},  // 0.1 L_inf perturbation is too visible in color
  'fgsmTargeted': {loss: 1},  // The 2nd loss function is too heavy for ImageNet
  'jsmaOnePixel': {ε: 75},  // This is unsuccessful. I estimate that it requires ~50x higher ε than CIFAR-10 to be successful on ImageNet, but that is too slow
  'cw': {κ: 5, c: 1, λ: 0.05}  // Generate higher confidence adversarial examples, and minimize distortion
};

/************************************************************************
* Load Datasets
************************************************************************/

/****************************** Load MNIST ******************************/

let mnistDataset;
let mnistUrl = 'data/mnist/mnist_sample.csv';
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

let cifarXUrl = 'data/cifar/cifar10_sample_x.json';
let cifarYUrl = 'data/cifar/cifar10_sample_y.json';

// Load data in form [{xs: x0_tensor, ys: y0_tensor}, {xs: x1_tensor, ys: y1_tensor}, ...]
let cifarX, cifarY, cifarDataset;
let loadingCifarX = fetch(cifarXUrl).then(res => res.json()).then(arr => cifarX = tf.data.array(arr).batch(1));
let loadingCifarY = fetch(cifarYUrl).then(res => res.json()).then(arr => cifarY = tf.data.array(arr).batch(1));
let loadingCifar = Promise.all([loadingCifarX, loadingCifarY]).then(() => tf.data.zip([cifarX, cifarY]).toArray()).then(ds => cifarDataset = ds.map(e => { return {xs: e[0], ys: e[1]}}));

/****************************** Load GTSRB ******************************/

let gtsrbXUrl = 'data/gtsrb/gtsrb_sample_x.json';
let gtsrbYUrl = 'data/gtsrb/gtsrb_sample_y.json';

// Load data in form [{xs: x0_tensor, ys: y0_tensor}, {xs: x1_tensor, ys: y1_tensor}, ...]
let gtsrbX, gtsrbY, gtsrbDataset;
let loadingGtsrbX = fetch(gtsrbXUrl).then(res => res.json()).then(arr => gtsrbX = tf.data.array(arr).batch(1));
let loadingGtsrbY = fetch(gtsrbYUrl).then(res => res.json()).then(arr => gtsrbY = tf.data.array(arr).batch(1));
let loadingGtsrb = Promise.all([loadingGtsrbX, loadingGtsrbY]).then(() => tf.data.zip([gtsrbX, gtsrbY]).toArray()).then(ds => gtsrbDataset = ds.map(e => { return {xs: e[0], ys: e[1]}}));

/****************************** Load ImageNet ******************************/

let imagenetXUrls = [
  '../data/imagenet/574_golf_ball.jpg',
  '../data/imagenet/217_english_springer.jpg',
  '../data/imagenet/701_parachute.jpg',
  '../data/imagenet/0_tench.jpg',
  '../data/imagenet/497_church.jpg',
  '../data/imagenet/566_french_horn.jpg'
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
    //drawImg(loadingImage, 'original');
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

let mnistXception;
let mnistResnet;
let mnistVgg16;
let mnistMobilenet;
async function loadMnistModel() {
  if (mnistVgg16 == undefined) { mnistVgg16 = await tf.loadGraphModel('data/mnist/vgg16/model.json'); }
  if (mnistResnet == undefined) { mnistResnet = await tf.loadGraphModel('data/mnist/resnet/model.json'); }
  if (mnistXception == undefined) { mnistXception = await tf.loadGraphModel('data/mnist/xception/model.json'); }
  if (mnistMobilenet == undefined) { mnistMobilenet = await tf.loadGrahpModel('data/mnist/mobilenet/model.json'); }
  //mnistModel = await tf.loadLayersModel('data/mnist/mnist_dnn.json');
}

/****************************** Load CIFAR-10 ******************************/

let cifarVgg16;
let cifarResnet;
let cifarXception;
let cifarMobilenet;
async function loadCifarModel() {
  if (cifarVgg16 == undefined) { cifarVgg16 = await tf.loadLayersModel('data/cifar/vgg16/model.json'); }
  if (cifarResnet == undefined) { cifarResnet = await tf.loadLayersModel('data/cifar/resnet/model.json'); }
  //if (cifarXception == undefined) { cifarXception = await tf.loadLayersModel('data/cifar/xception/model.json'); }
  if (cifarMobilenet == undefined) { cifarMobilenet = await tf.loadLayersModel('data/cifar/mobilenet/model.json'); }
}

/****************************** Load GTSRB ******************************/

let gtsrbModel;
let gtsrbVgg16;
let gtsrbResnet;
let gtsrbXception;
let gtsrbMobilenet;

async function loadGtsrbModel() {
  if (gtsrbModel !== undefined) { return; }
  gtsrbModel = await tf.loadLayersModel('data/gtsrb/gtsrb_cnn.json');
}

/****************************** Load ImageNet ******************************/

let imagenetModel;
let imagenetVgg16;
let imagenetResnet;
let imagenetXception;
let imagenetMobilenet;
async function loadImagenetModel() {
  if (imagenetVgg16 == undefined) { imagenetVgg16 = await tf.loadLayersModel('data/imagenet/vgg16/model.json'); }
  if (imagenetResnet == undefined) { imagenetResnet = await tf.loadLayersModel('data/imagenet/resnet/model.json'); }
  //if (imagenetXception == undefined) { imagenetXception = await tf.loadLayersModel('data/imagenet/xception/model.json'); }
  //if (imagenetMobilenet == undefined) { imagenetMobilenet = await tf.loadLayersModel('data/imagenet/mobilenet/model.json'); }
  if (imagenetMobilenet == undefined) { imagenetMobilenet = await mobilenet.load({version: 2, alpha: 1.0}); }
  
  
  imagenetMobilenet.predict = function (img) {
    return this.predictLogits(img).softmax();
  }
  imagenetMobilenet.predictLogits = function (img) {
    // Remove the first "background noise" logit
    // Copied from: https://github.com/tensorflow/tfjs-models/blob/708e3911fb01d0dfe70448acc3e8ca736fae82d3/mobilenet/src/index.ts#L232
    const logits1001 = this.model.predict(img);
    return logits1001.slice([0, 1], [-1, 1000]);
  }
  /*         Old Code for Mobilenet Imagnet Classifier
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
    */
}

/************************************************************************
* Attach Event Handlers
************************************************************************/

// On page load
window.addEventListener('load', showImage);
//window.addEventListener('load', resetAvailableAttacks);
//window.addEventListener('load', showBanners);

// Model selection dropdown
let architecture = "resnet"
export function changeArchitecture(arch){
	architecture = arch;
	showImage();
	resetOnNewImage();
	//resetAttack();
}

let dataset = "mnist"
export function changeDataset(ds){
	dataset = ds
	showImage();
	resetOnNewImage();
	//resetAttack();
}
//$('#select-model').addEventListener('change', showImage);
//$('#select-model').addEventListener('change', resetOnNewImage);
//$('#select-model').addEventListener('change', resetAttack);
//$('#select-model').addEventListener('change', removeLeftOverlay);

// Next image button
export function nextImage(){
	if (dataset === 'upload'){dataset = revertDataset;}
	showNextImage();
	resetOnNewImage();
	//resetAttack();
}
//$('#next-image').addEventListener('click', showNextImage);
//$('#next-image').addEventListener('click', resetOnNewImage);
//$('#next-image').addEventListener('click', resetAttack);

// Upload image button

let revertDataset;
export function uploadImage(){
	console.log("Stealing all your private data.");
	revertDataset = dataset;
	dataset = 'upload';
	getImg();
	resetOnNewImage();

	
	//showNextImage();
	//resetOnNewImage();
	//resetAttack();
}
// Predict button (original image
export function predictImg(){
    console.log("Releasing private data");
    predict();
    //removeTopRightOverlay();
}
//$('#predict-original').addEventListener('click', predict);
//$('#predict-original').addEventListener('click', removeTopRightOverlay);

// Target label dropdown
//$('#select-target').addEventListener('change', resetAttack);
let selectedTarget = 0;
export function changeTarget(target){
	selectedTarget =  parseInt(target);
}

// Attack algorithm dropdown
//$('#select-attack').addEventListener('change', resetAttack);
let selectedAttack;
export function changeAttack(attack){
	selectedAttack = attack;
}

// Generate button
let flag = true;
export function attack(){
    console.log("Destroying all familiarity");
	if(flag){
		generateAdv();
		flag = false;
	}
	else{
		predictAdv();
		flag = true;
	}
    //removeTopRightOverlay();
}
//$('#generate-adv').addEventListener('click', generateAdv);
//$('#generate-adv').addEventListener('click', removeBottomRightOverlay);

// Predict button (adversarial image)
//$('#predict-adv').addEventListener('click', predictAdv);

// View noise / view image link
//$('#view-noise').addEventListener('click', viewNoise);
//$('#view-image').addEventListener('click', viewImage);

/************************************************************************
* Define Event Handlers
************************************************************************/

/**
 * Gets image uploaded by the user. 
 */
let loadedUpload;
async function getImg(){
	const input = document.getElementById("fileid");
	let source = input.files[0];
	
	let loadingUpload= [];
	document.getElementsByClassName("upload_img").forEach(e => {
		loadingUpload.push(loadImage(e, URL.createObjectURL(source)));
	});
	
	let loadedUploadData = Promise.all(loadingUpload);
	
	await loadedUploadData.then(() => {
		let img = document.getElementsByClassName("upload_img")[0];
		console.log(tf.browser.fromPixels(img).div(255.0).reshape([1, 224, 224, 3]));
		loadedUpload = tf.browser.fromPixels(img).div(255.0).reshape([1, 224, 224, 3]);
	});
	
	drawImg(loadedUpload, "original");
}

/**
 * Renders the next image from the sample dataset in the original canvas
 */
function showNextImage() {
  let modelName = dataset;
  //let modelName = $('#select-model').value;
  if (modelName === 'mnist') { showNextMnist(); }
  else if (modelName === 'cifar') { showNextCifar(); }
  else if (modelName === 'gtsrb') { showNextGtsrb(); }
  else if (modelName === 'imagenet') { showNextImagenet(); }
}

/**
 * Renders the current image from the sample dataset in the original canvas
 */
function showImage() {
  let modelName = dataset;
  //let modelName = $('#select-model').value;
  if (modelName === 'mnist') { showMnist(); }
  else if (modelName === 'cifar') { showCifar(); }
  else if (modelName === 'gtsrb') { showGtsrb(); }
  else if (modelName === 'imagenet') { showImagenet(); }
}

export function testResponse(value){
	let response = "Confirmation of Event from " + value;
	console.log(response);
}
/**
 * Computes & displays prediction of the current original image
 */
async function predict() {
  //$('#predict-original').disabled = true;
  //$('#predict-original').innerText = 'Loading...';

  let model;
  if (dataset === 'mnist') {
    
    console.log(architecture);
    await loadMnistModel();
    await loadingMnist;    
    
    if (architecture === 'resnet') { model = mnistResnet; }
    else if (architecture === 'vgg16') {model = mnistVgg16; }
    else if (architecture === 'xception') {model = mnistXception; }
    else if (architecture === 'mobilenet') {model = mnistMobilenet; }
    
    let lblIdx = mnistDataset[mnistIdx].ys.argMax(1).dataSync()[0];

    let img = mnistDataset[mnistIdx].xs;
    let resizedImg = tf.image.resizeNearestNeighbor(img.reshape([1, 28, 28, 1]), [32, 32]);
    let RGB = tf.image.grayscaleToRGB(resizedImg);
    //console.log(model)
    //_predict(mnistModel, mnistDataset[mnistIdx].xs, lblIdx, MNIST_CLASSES);
    _predict(model, RGB, lblIdx, MNIST_CLASSES);
  } else if (dataset === 'cifar') {
    await loadCifarModel();
    await loadingCifar;
    
    if (architecture === 'resnet') { model = cifarResnet; }
    else if (architecture === 'vgg16') {model = cifarVgg16; }
    else if (architecture === 'xception') {model = cifarXception; }
    else if (architecture === 'mobilenet') {model = cifarMobilenet; }
    
    let lblIdx = cifarDataset[cifarIdx].ys.argMax(1).dataSync()[0];
    _predict(model, cifarDataset[cifarIdx].xs, lblIdx, CIFAR_CLASSES);
  } else if (dataset === 'gtsrb') {
    await loadGtsrbModel();
    await loadingGtsrb;
    
    if (architecture === 'resnet') { model = gtsrbResnet; }
    else if (architecture === 'vgg16') {model = gtsrbVgg16; }
    else if (architecture === 'xception') {model = gtsrbXception; }
    else if (architecture === 'mobilenet') {model = gtsrbMobilenet; }
    
    let lblIdx = gtsrbDataset[gtsrbIdx].ys.argMax(1).dataSync()[0];
    _predict(model, gtsrbDataset[gtsrbIdx].xs, lblIdx, GTSRB_CLASSES);
  } else if (dataset === 'imagenet') {
    await loadImagenetModel();
    await loadedImagenetData;
    
	console.log(architecture);
    if (architecture === 'resnet') { model = imagenetResnet; }
    else if (architecture === 'vgg16') {model = imagenetVgg16; }
    else if (architecture === 'xception') {model = imagenetXception; }
    else if (architecture === 'mobilenet') {model = imagenetMobilenet; }
    
	console.log(imagenetIdx);
	console.log(imagenetX[imagenetIdx].shape);
	
    _predict(model, imagenetX[imagenetIdx], imagenetYLbls[imagenetIdx], IMAGENET_CLASSES);
  } else if (dataset === 'upload') {
    await loadImagenetModel();
    await loadedImagenetData;
    
	console.log(architecture);
    if (architecture === 'resnet') { model = imagenetResnet; }
    else if (architecture === 'vgg16') {model = imagenetVgg16; }
    else if (architecture === 'xception') {model = imagenetXception; }
    else if (architecture === 'mobilenet') {model = imagenetMobilenet; }
	
    _predict(model, loadedUpload, 'upload', IMAGENET_CLASSES);
  }

  //$('#predict-original').innerText = 'Run Neural Network';

  function _predict(model, img, lblIdx, CLASS_NAMES) {
    // Generate prediction
    let pred = model.predict(img);
    //console.log(pred.dataSync())
    console.log(pred.max().dataSync())
    let predLblIdx = pred.argMax(1).dataSync()[0];
    let predProb = pred.max().dataSync()[0];

    // Display prediction
    let status = {msg: '✅ Prediction is Correct.', statusClass: 'status-green'};  // Predictions on the sample should always be correct
    showPrediction(`Prediction: "${CLASS_NAMES[predLblIdx]}"<br/>Probability: ${(predProb * 100).toFixed(2)}%`, status);
  }
 }

/**
 * Generates adversarial example from the current original image
 */
let advPrediction, advStatus;
async function generateAdv() {
  //$('#generate-adv').disabled = true;
  //$('#generate-adv').innerText = 'Loading...';

  let attack;
  switch (selectedAttack) {
    case 'fgsmTargeted': attack = fgsmTargeted; break;
    case 'bimTargeted': attack = bimTargeted; break;
    case 'jsmaOnePixel': attack = jsmaOnePixel; break;
    case 'jsma': attack = jsma; break;
    case 'cw': attack = cw; break;
  }
    
  let adv_model;
  let modelName = dataset;
  let targetLblIdx = selectedTarget;
  if (dataset === 'mnist') {
    await loadMnistModel();
    await loadingMnist;
    
    if (architecture === 'resnet') { adv_model = mnistResnet; }
    else if (architecture === 'vgg16') {adv_model = mnistVgg16; }
    else if (architecture === 'xception') {adv_model = mnistXception; }
    else if (architecture === 'mobilenet') {adv_model = mnistMobilenet; }
	
	let img = mnistDataset[mnistIdx].xs;
    let resizedImg = tf.image.resizeNearestNeighbor(img.reshape([1, 28, 28, 1]), [32, 32]);
    let RGB = tf.image.grayscaleToRGB(resizedImg);
    
    await _generateAdv(adv_model, RGB, mnistDataset[mnistIdx].ys, MNIST_CLASSES, MNIST_CONFIGS[attack.name]);
  } else if (dataset === 'cifar') {
    await loadCifarModel();
    await loadingCifar;
    
    if (architecture === 'resnet') { adv_model = cifarResnet; }
    else if (architecture === 'vgg16') {adv_model = cifarVgg16; }
    else if (architecture === 'xception') {adv_model = cifarXception; }
    else if (architecture === 'mobilenet') {adv_model = cifarMobilenet; }
    
    await _generateAdv(adv_model, cifarDataset[cifarIdx].xs, cifarDataset[cifarIdx].ys, CIFAR_CLASSES, CIFAR_CONFIGS[attack.name]);
  } else if (dataset === 'gtsrb') {
    await loadGtsrbModel();
    await loadingGtsrb;
    
    if (architecture === 'resnet') { adv_model = gtsrbResnet; }
    else if (architecture === 'vgg16') {adv_model = gtsrbVgg16; }
    else if (architecture === 'xception') {adv_model = gtsrbXception; }
    else if (architecture === 'mobilenet') {adv_model = gtsrbMobilenet; }
    
    await _generateAdv(adv_model, gtsrbDataset[gtsrbIdx].xs, gtsrbDataset[gtsrbIdx].ys, GTSRB_CLASSES, GTSRB_CONFIGS[attack.name]);
  } else if (dataset === 'imagenet') {
    await loadImagenetModel();
    await loadedImagenetData;
    
    if (architecture === 'resnet') { adv_model = imagenetResnet; }
    else if (architecture === 'vgg16') {adv_model = imagenetVgg16; }
    else if (architecture === 'xception') {adv_model = imagenetXception; }
    else if (architecture === 'mobilenet') {adv_model = imagenetMobilenet; }
    
    await _generateAdv(adv_model, imagenetX[imagenetIdx], imagenetY[imagenetIdx], IMAGENET_CLASSES, IMAGENET_CONFIGS[attack.name]);
  } else if (dataset === 'upload') {
    await loadImagenetModel();
    await loadedImagenetData;
    
    if (architecture === 'resnet') { adv_model = imagenetResnet; }
    else if (architecture === 'vgg16') {adv_model = imagenetVgg16; }
    else if (architecture === 'xception') {adv_model = imagenetXception; }
    else if (architecture === 'mobilenet') {adv_model = imagenetMobilenet; }
    
    await _generateAdv(adv_model, loadedUpload, imagenetY[0], IMAGENET_CLASSES, IMAGENET_CONFIGS[attack.name]);
  }

  //$('#latency-msg').style.display = 'none';
  //$('#generate-adv').innerText = 'Generate';
  //$('#predict-adv').innerText = 'Run Neural Network';
  //$('#predict-adv').disabled = false;

  async function _generateAdv(model, img, lbl, CLASS_NAMES, CONFIG) {
    // Generate adversarial example
    let targetLbl = tf.oneHot(targetLblIdx, lbl.shape[1]).reshape(lbl.shape);
    let aimg = tf.tidy(() => attack(model, img, lbl, targetLbl, CONFIG));

    // Display adversarial example
    //$('#difference').style.display = 'block';
    await drawImg(aimg, 'adversarial');

    // Compute & store adversarial prediction
    let pred = model.predict(aimg);
    let predLblIdx = pred.argMax(1).dataSync()[0];
    let predProb = pred.max().dataSync()[0];
    advPrediction = `Prediction: "${CLASS_NAMES[predLblIdx]}"<br/>Probability: ${(predProb * 100).toFixed(2)}%`;

    // Compute & store attack success/failure message
    let lblIdx = lbl.argMax(1).dataSync()[0];
    if (predLblIdx === targetLblIdx) {
      advStatus = {msg: '❌ Prediction is wrong. Attack succeeded!', statusClass: 'status-red'};
    } else if (predLblIdx !== lblIdx) {
      advStatus = {msg: '❌ Prediction is wrong. Attack partially succeeded!', statusClass: 'status-orange'};
    } else {
      advStatus = {msg: '✅ Prediction is still correct. Attack failed.', statusClass: 'status-green'};
    }

	console.log(advStatus);
    // Also compute and draw the adversarial noise (hidden until the user clicks on it)
    let noise = tf.sub(aimg, img).add(0.5).clipByValue(0, 1);  // [Szegedy 14] Intriguing properties of neural networks
    drawImg(noise, 'adversarial-noise');
	console.log("Adversified");
  }
}

/**
 * Displays prediction for the current adversarial image
 * (This function just renders the status we've already computed in generateAdv())
 */
function predictAdv() {
  //$('#predict-adv').disabled = true;
  showAdvPrediction(advPrediction, advStatus);
}

/**
 * Show adversarial noise when the user clicks on the "view noise" link
 */
async function viewNoise() {
  $('#difference').style.display = 'none';
  $('#difference-noise').style.display = 'block';
  $('#adversarial').style.display = 'none';
  $('#adversarial-noise').style.display = 'block';
}

/**
 * Show adversarial image when the user clicks on the "view image" link
 */
async function viewImage() {
  $('#difference').style.display = 'block';
  $('#difference-noise').style.display = 'none';
  $('#adversarial').style.display = 'block';
  $('#adversarial-noise').style.display = 'none';
}

/**
 * Reset entire dashboard UI when a new image is selected
 */
function resetOnNewImage() {
  //$('#predict-original').disabled = false;
  //$('#predict-original').innerText = 'Run Neural Network';
  $('#prediction').style.display = 'none';
  $('#prediction-status').innerHTML = '';
  $('#prediction-status').className = '';
  $('#prediction-status').style.marginBottom = '9px';
  //resetAttack();
  //resetAvailableAttacks();
}

/**
 * Reset attack UI when a new target label, attack, or image is selected
 */
async function resetAttack() {
  $('#generate-adv').disabled = false;
  $('#predict-adv').disabled = true;
  $('#predict-adv').innerText = 'Click "Generate" First';
  $('#difference').style.display = 'none';
  $('#difference-noise').style.display = 'none';
  $('#prediction-adv').style.display = 'none';
  $('#prediction-adv-status').innerHTML = '';
  $('#prediction-adv-status').className = '';
  $('#prediction-adv-status').style.marginBottom = '9px';
  await drawImg(tf.ones([1, 224, 224, 1]), 'adversarial');
  await drawImg(tf.ones([1, 224, 224, 1]), 'adversarial-noise');
  $('#adversarial').style.display = 'block';
  $('#adversarial-noise').style.display = 'none';

  if ($('#select-model').value === 'gtsrb' || $('#select-model').value === 'imagenet') {
    $('#latency-msg').style.display = 'block';
  } else {
    $('#latency-msg').style.display = 'none';
  }
}

/**
 * Reset available attacks and target labels when a new image is selected
 */
function resetAvailableAttacks() {
  const MNIST_TARGETS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const CIFAR_TARGETS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const GTSRB_TARGETS = [8, 0, 14, 17];
  const IMAGENET_TARGETS = [934, 413, 151];

  let modelName = 'mnist';
  //let modelName = $('#select-model').value;
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

/**
 * Removes the overlay on the left half of the dashboard when the user selects a model
 */
function removeLeftOverlay() {
  $('#original-image-overlay').style.display = 'none';
  $('#original-canvas-overlay').style.display = 'none';
  $('#original-prediction-overlay').style.display = 'none';
}

/**
 * Removes the overlay on the top right of the dashboard when the user makes their first prediction
 */
function removeTopRightOverlay() {
  $('#adversarial-image-overlay').style.display = 'none';
  $('#adversarial-canvas-overlay').style.display = 'none';
}

/**
 * Removes the overlay on the bottom right of the dashboard when the user generates an adversarial example
 */
function removeBottomRightOverlay() {
  $('#adversarial-prediction-overlay').style.display = 'none';
}

/**
 * Check the user's device and display appropriate warning messages
 */
function showBanners() {
  if (!supports32BitWebGL()) { $('#mobile-banner').style.display = 'block'; }
  else if (!isDesktopChrome()) { $('#chrome-banner').style.display = 'block'; }
}

/**
 * Returns if it looks like the user is on desktop Google Chrome
 * https://stackoverflow.com/a/13348618/908744
 */
function isDesktopChrome() {
  var isChromium = window.chrome;
  var winNav = window.navigator;
  var vendorName = winNav.vendor;
  var isOpera = typeof window.opr !== "undefined";
  var isIEedge = winNav.userAgent.indexOf("Edge") > -1;
  var isIOSChrome = winNav.userAgent.match("CriOS");

  if (isIOSChrome) {
    return false;
  } else if (
    isChromium !== null &&
    typeof isChromium !== "undefined" &&
    vendorName === "Google Inc." &&
    isOpera === false &&
    isIEedge === false
  ) {
    return true;
  } else {
    return false;
  }
}

/**
 * Returns if the current device supports WebGL 32-bit
 * https://www.tensorflow.org/js/guide/platform_environment#precision
 */
function supports32BitWebGL() {
  return tf.ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE') && tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED');
}

/************************************************************************
* Visualize Images
************************************************************************/

function showPrediction(msg, status) {
  console.log("predicting, no writing");
  $('#prediction').innerHTML = msg;
  $('#prediction').style.display = 'block';
  $('#prediction-status').innerHTML = status.msg;
  $('#prediction-status').className = status.statusClass;
  $('#prediction-status').style.marginBottom = '15px';
}

function showAdvPrediction(msg, status) {
  $('#prediction-adv').innerHTML = msg;
  $('#prediction-adv').style.display = 'block';
  $('#prediction-adv-status').innerHTML = status.msg;
  $('#prediction-adv-status').className = status.statusClass;
  $('#prediction-adv-status').style.marginBottom = '15px';
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

async function drawImg(img, element) {
  // Draw image
  let canvas = document.getElementById(element);
  if (img.shape[0] === 1) { img = img.squeeze(0); }
  if (img.shape[0] === 784) {
    let resizedImg = tf.image.resizeNearestNeighbor(img.reshape([28, 28, 1]), [224, 224]);
    await tf.browser.toPixels(resizedImg, canvas);
  } else {
    let resizedImg = tf.image.resizeNearestNeighbor(img, [224, 224]);
    await tf.browser.toPixels(resizedImg, canvas);
  }
}
/* eslint-enable no-unused-vars */