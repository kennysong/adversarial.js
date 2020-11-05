/************************************************************************
* Load Dataset
************************************************************************/

let xUrl = 'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/gtsrb_sample_x_4.json?alt=media';
let yUrl = 'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/gtsrb_sample_y_4.json?alt=media';

// Load data in form [{xs: x0_tensor, ys: y0_tensor}, {xs: x1_tensor, ys: y1_tensor}, ...]
let x, y, dataset;
let loadingX = fetch(xUrl).then(res => res.json()).then(arr => x = tf.data.array(arr).batch(1));
let loadingY = fetch(yUrl).then(res => res.json()).then(arr => y = tf.data.array(arr).batch(1));
let loadingData = Promise.all([loadingX, loadingY]).then(() => tf.data.zip([x, y]).toArray()).then(ds => dataset = ds.map(e => { return {xs: e[0], ys: e[1]}}));

/************************************************************************
* Load Model
************************************************************************/

let model;
let loadingModel = tf.loadLayersModel('https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/gtsrb_cnn_3.json?alt=media')
  .then(m => model = m);

let allLoaded = Promise.all([loadingData, loadingModel]);

/************************************************************************
* Visualize Attacks
************************************************************************/

const CONFIGS = {
  'bimTargeted': {iters: 50},  // Needs more iterations to work well
  'jsmaOnePixel': {ε: 75},  // Works well with the same settings as CIFAR-10
};

const CLASS_NAMES = {
  // Adapted from: https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/signnames.csv
  0: "20km/hr",
  1: "30km/hr",
  2: "50km/hr",
  3: "60km/hr",
  4: "70km/hr",
  5: "80km/hr",
  6: "End of 80km/hr",
  7: "100km/hr",
  8: "120km/hr",
  9: "No passing",
  10: "No passing for heavy vehicles",
  11: "Right-of-way",
  12: "Priority road",
  13: "Yield",
  14: "Stop Sign",
  15: "No vehicles",
  16: "No heavy vehicles",
  17: "Do Not Enter",
  18: "General caution",
  19: "Dangerous left curve",
  20: "Dangerous right curve",
  21: "Double curve",
  22: "Bumpy road",
  23: "Slippery road",
  24: "Right narrows",
  25: "Road work",
  26: "Traffic signals",
  27: "Pedestrians",
  28: "Children crossing",
  29: "Bicycles crossing",
  30: "Beware of ice/snow",
  31: "Wild animals crossing",
  32: "End speed limits",
  33: "Turn right ahead",
  34: "Turn left ahead",
  35: "Ahead only",
  36: "Go straight or right",
  37: "Go straight or left",
  38: "Keep right",
  39: "Keep left",
  40: "Roundabout mandatory",
  41: "End of no passing",
  42: "End of no passing by heavy vehicles",
};
// For display, truncate class names to <20 characters
for (let i = 0; i < 42; i++) {
  if (CLASS_NAMES[i].length > 20) {
      CLASS_NAMES[i] = CLASS_NAMES[i].slice(0, 20);
  }
}

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
    await drawImg(img, i.toString(), attack.name, `Prediction: ${CLASS_NAMES[lblIdx]}<br/>Prob: ${p.toFixed(3)}`);

    // Generate adversarial image from attack
    let aimg = tf.tidy(() => attack(model, img, lbl, CONFIGS[attack.name]));

    // Display adversarial image and its probability
    p = model.predict(aimg).max(1).dataSync()[0];
    let albl = model.predict(aimg).argMax(1).dataSync()[0];
    if (albl !== lblIdx) {
      successes++;
      await drawImg(aimg, `${i}a`, attack.name, `Prediction: ${CLASS_NAMES[albl]}<br/>Prob: ${p.toFixed(3)}`, true);
    } else {
      await drawImg(aimg, `${i}a`, attack.name, `Prediction: ${CLASS_NAMES[albl]}<br/>Prob: ${p.toFixed(3)}`);
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
    await drawImg(img, i.toString(), attack.name, `Prediction: ${CLASS_NAMES[lblIdx]}<br/>Prob: ${p.toFixed(3)}`);

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
        await drawImg(aimg, `${i}${j}`, attack.name, `Prediction: ${CLASS_NAMES[targetLblIdx]}<br/>Prob: ${p.toFixed(3)}`, true);
      } else {
        await drawImg(aimg, `${i}${j}`, attack.name, `Prediction: ${CLASS_NAMES[targetLblIdx]}<br/>Prob: ${p.toFixed(3)}`);
      }
    }
  }

  document.getElementById(`${attack.name}-success-rate`).innerText = `Success rate: ${(successes / (NUM_ROWS*NUM_COLS-NUM_COLS)).toFixed(2)}`;
}