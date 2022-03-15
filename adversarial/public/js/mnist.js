/************************************************************************
* Load Model
************************************************************************/

let model;
let loadingModel = tf.loadLayersModel('data/mnist/mnist_dnn.json')
  .then(m => { model = m });

/************************************************************************
* Load Dataset
************************************************************************/

let dataset;
let dataUrl = 'data/mnist/mnist_sample.csv';
let loadingDataset = tf.data.csv(dataUrl, {columnConfigs: {label: {isLabel: true}}})
  .take(10).map(({xs, ys}) => {
    xs = Object.values(xs).map(e => e/255);  // Convert from feature object to array, and normalize
    ys = tf.oneHot(Object.values(ys), 10).squeeze();  // Convert from feature object to scalar, and turn into one-hot vector
    return {xs: xs, ys: ys};
  })
  .batch(1)
  .toArray()
  .then(ds => dataset = ds);

let allLoaded = Promise.all([loadingDataset, loadingModel]);

/************************************************************************
* Visualize Attacks
************************************************************************/

const CONFIGS = {
  'fgsmTargeted': {ε: 0.2},  // Targeted FGSM works slightly better on MNIST with higher distortion
  'bimTargeted': {iters: 20},  // Targeted BIM works slightly better on MNIST with more iterations (pushes misclassification confidence up)
};

async function drawImg(img, element, attackName, msg, success) {
  let canvas = document.getElementById(attackName).getElementsByClassName(element)[0];
  let resizedImg = tf.image.resizeNearestNeighbor(img.reshape([28, 28, 1]), [56, 56]);
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

  for (let i = 0; i < 10; i++) {  // For each row
    let img = dataset[i].xs;
    let lbl = dataset[i].ys;

    // Compute and display original class probability
    let p = model.predict(img).dataSync()[i];
    await drawImg(img, i.toString(), attack.name, `Pred: ${i}<br/>Prob: ${p.toFixed(3)}`);

    // Generate adversarial image from attack
    let aimg = tf.tidy(() => attack(model, img, lbl, CONFIGS[attack.name]));

    // Display adversarial image and its probability
    p = model.predict(aimg).max(1).dataSync()[0];
    let albl = model.predict(aimg).argMax(1).dataSync()[0];
    let oldlbl = lbl.argMax(1).dataSync()[0];
    if (albl !== oldlbl) {
      successes++;
      await drawImg(aimg, `${i}a`, attack.name, `Pred: ${albl}<br/>Prob: ${p.toFixed(3)}`, true);
    } else {
      await drawImg(aimg, `${i}a`, attack.name, `Pred: ${albl}<br/>Prob: ${p.toFixed(3)}`);
    }
  }

  document.getElementById(`${attack.name}-success-rate`).innerText = `Success rate: ${(successes / 10).toFixed(1)}`;
}

export async function runTargeted(attack) {
  await allLoaded;
  let successes = 0;

  for (let i = 0; i < 10; i++) {  // For each row
    let img = dataset[i].xs;
    let lbl = dataset[i].ys;

    // Compute and display original class probability
    let p = model.predict(img).dataSync()[i];
    await drawImg(img, i.toString(), attack.name, `Pred: ${i}<br/>Prob: ${p.toFixed(3)}`);

    for (let j = 0; j < 10; j++) {  // For each target label
      // Draw a black square if the target class is just the original class
      if (j === lbl.argMax(1).dataSync()[0]) {
        await drawImg(tf.zerosLike(img), `${i}${j}`, attack.name);
        continue;
      }

      // Generate adversarial image from attack
      let targetLbl = tf.oneHot(j, 10).reshape([1, 10]);
      let aimg = tf.tidy(() => attack(model, img, lbl, targetLbl, CONFIGS[attack.name]));

      // Display adversarial image and its probability
      p = model.predict(aimg).dataSync()[j];
      let predLbl = model.predict(aimg).argMax(1).dataSync()[0];
      if (predLbl === j) {
        successes++;
        await drawImg(aimg, `${i}${j}`, attack.name, `Pred: ${j}<br/>Prob: ${p.toFixed(3)}`, true);
      } else {
        await drawImg(aimg, `${i}${j}`, attack.name, `Pred: ${j}<br/>Prob: ${p.toFixed(3)}`);
      }
    }
  }

  document.getElementById(`${attack.name}-success-rate`).innerText = `Success rate: ${(successes / 90).toFixed(2)}`;
}