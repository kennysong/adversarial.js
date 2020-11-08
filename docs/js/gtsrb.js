import {GTSRB_CLASSES} from './class_names.js';

/************************************************************************
* Load Dataset
************************************************************************/

let xUrl = 'data/gtsrb/gtsrb_sample_x.json';
let yUrl = 'data/gtsrb/gtsrb_sample_y.json';

// Load data in form [{xs: x0_tensor, ys: y0_tensor}, {xs: x1_tensor, ys: y1_tensor}, ...]
let x, y, dataset;
let loadingX = fetch(xUrl).then(res => res.json()).then(arr => x = tf.data.array(arr).batch(1));
let loadingY = fetch(yUrl).then(res => res.json()).then(arr => y = tf.data.array(arr).batch(1));
let loadingData = Promise.all([loadingX, loadingY]).then(() => tf.data.zip([x, y]).toArray()).then(ds => dataset = ds.map(e => { return {xs: e[0], ys: e[1]}}));

/************************************************************************
* Load Model
************************************************************************/

let model;
let loadingModel = tf.loadLayersModel('data/gtsrb/gtsrb_cnn.json')
  .then(m => model = m);

let allLoaded = Promise.all([loadingData, loadingModel]);

/************************************************************************
* Visualize Attacks
************************************************************************/

const CONFIGS = {
  'bimTargeted': {iters: 50},  // Needs more iterations to work well
  'jsmaOnePixel': {ε: 75},  // Works well with the same settings as CIFAR-10
};

async function drawImg(img, element, attackName, msg, success) {
  let canvas = document.getElementById(attackName).getElementsByClassName(element)[0];
  let resizedImg = tf.image.resizeNearestNeighbor(img.reshape([64, 64, 3]), [128, 128]);
  await tf.browser.toPixels(resizedImg, canvas);

  if (msg !== undefined) {
    canvas.nextSibling.innerHTML = msg;
  }
  if (success === true) {
    canvas.style.borderColor = 'lime';
    canvas.style.borderWidth = '2px';
  }
}

export async function runUntargeted(attack) {
  await allLoaded;
  let successes = 0;

  let NUM_ROWS = 4;

  for (let i = 0; i < NUM_ROWS; i++) {  // For each row
    let img = dataset[i].xs;
    let lbl = dataset[i].ys;
    let lblIdx = lbl.argMax(1).dataSync()[0];

    // Compute and display original class probability
    let p = model.predict(img).dataSync()[lblIdx];
    await drawImg(img, i.toString(), attack.name, `Prediction: ${GTSRB_CLASSES[lblIdx]}<br/>Prob: ${p.toFixed(3)}`);

    // Generate adversarial image from attack
    let aimg = tf.tidy(() => attack(model, img, lbl, CONFIGS[attack.name]));

    // Display adversarial image and its probability
    p = model.predict(aimg).max(1).dataSync()[0];
    let albl = model.predict(aimg).argMax(1).dataSync()[0];
    if (albl !== lblIdx) {
      successes++;
      await drawImg(aimg, `${i}a`, attack.name, `Prediction: ${GTSRB_CLASSES[albl]}<br/>Prob: ${p.toFixed(3)}`, true);
    } else {
      await drawImg(aimg, `${i}a`, attack.name, `Prediction: ${GTSRB_CLASSES[albl]}<br/>Prob: ${p.toFixed(3)}`);
    }
  }

  document.getElementById(`${attack.name}-success-rate`).innerText = `Success rate: ${(successes / 4).toFixed(1)}`;
}

export async function runTargeted(attack) {
  await allLoaded;
  let successes = 0;
  let targetLblIdxs = [14, 17, 0, 8];

  let NUM_ROWS = 4;
  let NUM_COLS = targetLblIdxs.length;

  for (let i = 0; i < NUM_ROWS; i++) {  // For each row
    let img = dataset[i].xs;
    let lbl = dataset[i].ys;
    let lblIdx = lbl.argMax(1).dataSync()[0];

    // Compute and display original class probability
    let p = model.predict(img).dataSync()[lblIdx];
    await drawImg(img, i.toString(), attack.name, `Prediction: ${GTSRB_CLASSES[lblIdx]}<br/>Prob: ${p.toFixed(3)}`);

    for (let j = 0; j < NUM_COLS; j++) {  // For each target label
      let targetLblIdx = targetLblIdxs[j];
      let targetLbl = tf.oneHot(targetLblIdx, 43).reshape([1, 43]);

      // Draw a black square if the target class is just the original class
      if (targetLblIdx === lblIdx) {
        await drawImg(tf.zerosLike(img), `${i}${j}`, attack.name);
        continue;
      }

      // Generate adversarial image from attack
      let aimg = tf.tidy(() => attack(model, img, lbl, targetLbl, CONFIGS[attack.name]));

      // Display adversarial image and its probability
      p = model.predict(aimg).dataSync()[targetLblIdx];
      let predLbl = model.predict(aimg).argMax(1).dataSync()[0];
      if (predLbl === targetLblIdx) {
        successes++;
        await drawImg(aimg, `${i}${j}`, attack.name, `Prediction: ${GTSRB_CLASSES[targetLblIdx]}<br/>Prob: ${p.toFixed(3)}`, true);
      } else {
        await drawImg(aimg, `${i}${j}`, attack.name, `Prediction: ${GTSRB_CLASSES[targetLblIdx]}<br/>Prob: ${p.toFixed(3)}`);
      }
    }
  }

  document.getElementById(`${attack.name}-success-rate`).innerText = `Success rate: ${(successes / (NUM_ROWS*NUM_COLS-NUM_COLS)).toFixed(2)}`;
}