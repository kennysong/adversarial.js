/************************************************************************
* Load Dataset
************************************************************************/

let x, y;

function loadData() {
  let csvUrl = 'data/mnist/mnist_test.csv';
  let csvDataset = tf.data.csv(csvUrl, {columnConfigs: {label: {isLabel: true}}});

  return loadingData = csvDataset.map(({xs, ys}) => {
    xs = Object.values(xs).map(e => e/255);  // Convert from feature object to array, and normalize
    ys = tf.oneHot(Object.values(ys), 10).squeeze();  // Convert from feature object to scalar, and turn into one-hot vector
    return {xs: xs, ys: ys};
  })
  .batch(10000)
  .toArray()
  .then(ds => { x = ds[0].xs; y = ds[0].ys; });
}

/************************************************************************
* Load Model
************************************************************************/

let model;

function loadModel() {
  return tf.loadLayersModel('data/mnist/mnist_dnn.json')
  .then(m => { model = m });
}

/************************************************************************
* Evaluate Model
************************************************************************/

document.getElementById('start-evaluation').addEventListener("click", () => {
  document.getElementById('start-evaluation').style.display = 'none';

  addStatus('\nLoading model and data...');
  let loadingModel = loadModel();
  let loadingData = loadData();

  Promise.all([loadingData, loadingModel]).then(() => {
    addStatus('Loaded model and data.');
    addStatus('Evaluating model...');

    let yPred = model.predict(x);
    let acc = tf.metrics.categoricalAccuracy(y, yPred).mean();
    let loss = tf.metrics.categoricalCrossentropy(y, yPred).mean();

    addStatus('Evaluated model.\n')
    addStatus(`Test Set Accuracy: ${acc.dataSync()[0]}`);
    addStatus(`Test Set Loss: ${loss.dataSync()[0]}`);
  });
});

/************************************************************************
* Useful snippets
************************************************************************/

function addStatus(msg) {
  document.getElementById('status').innerText += '\n' + msg;
}