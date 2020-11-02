/************************************************************************
* Load Model
************************************************************************/

let model;
let loadingModel = tf.loadLayersModel('https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/mnist_dnn.json?alt=media')
  .then(m => { model = m });

/************************************************************************
* Load Dataset
************************************************************************/

let dataset;
let dataUrl = 'https://storage.googleapis.com/download/storage/v1/b/kennysong-mnist/o/mnist_ten.csv?alt=media';
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

    let p = model.predict(img).dataSync()[i];
    await drawImg(img, i.toString(), attack.name, `Class: ${i}<br/>Prob: ${p.toFixed(3)}`);

    let aimg = tf.tidy(() => attack(model, img, lbl));

    p = model.predict(aimg).max(1).dataSync()[0];
    let albl = model.predict(aimg).argMax(1).dataSync()[0];
    let oldlbl = lbl.argMax(1).dataSync()[0];
    if (albl !== oldlbl) {
      successes++;
      await drawImg(aimg, `${i}a`, attack.name, `Class: ${albl}<br/>Prob: ${p.toFixed(3)}`, true);
    }
    await drawImg(aimg, `${i}a`, attack.name, `Class: ${albl}<br/>Prob: ${p.toFixed(3)}`);
  }

  document.getElementById(`${attack.name}-success-rate`).innerText = `Success rate: ${(successes / 10).toFixed(1)}`;
}

export async function runTargeted(attack) {
  await allLoaded;
  let successes = 0;

  for (let i = 0; i < 10; i++) {  // For each row
    let img = dataset[i].xs;
    let lbl = dataset[i].ys;

    let p = model.predict(img).dataSync()[i];
    await drawImg(img, i.toString(), attack.name, `Class: ${i}<br/>Prob: ${p.toFixed(3)}`);

    for (let j = 0; j < 10; j++) {  // For each target label
      if (j === lbl.argMax(1).dataSync()[0]) {
        await drawImg(tf.zerosLike(img), `${i}${j}`, attack.name);
        continue;
      }

      let targetLbl = tf.oneHot(j, 10).reshape([1, 10]);
      let aimg = tf.tidy(() => attack(model, img, lbl, targetLbl));

      p = model.predict(aimg).dataSync()[j];
      let predLbl = model.predict(aimg).argMax(1).dataSync()[0];
      if (predLbl === j) {
        successes++;
        await drawImg(aimg, `${i}${j}`, attack.name, `Class: ${j}<br/>Prob: ${p.toFixed(3)}`, true);
      } else {
        await drawImg(aimg, `${i}${j}`, attack.name, `Class: ${j}<br/>Prob: ${p.toFixed(3)}`);
      }
    }
  }

  document.getElementById(`${attack.name}-success-rate`).innerText = `Success rate: ${(successes / 90).toFixed(2)}`;
}