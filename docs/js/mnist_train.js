/************************************************************************
* Load Dataset
************************************************************************/

let csvUrl = 'data/mnist/mnist_train.csv';
let csvDataset = tf.data.csv(csvUrl, {columnConfigs: {label: {isLabel: true}}});

let flattenedDataset = csvDataset.map(({xs, ys}) => {
    xs = Object.values(xs).map(e => e/255);  // Convert from feature object to array, and normalize
    ys = tf.oneHot(Object.values(ys), 10).squeeze();  // Convert from feature object to scalar, and turn into one-hot vector
    return {xs: xs, ys: ys};
  })
 .batch(512);

/************************************************************************
* Define & Train Model
************************************************************************/

let model = tf.sequential();
model.add(tf.layers.dense({inputShape: [784], units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: 10}));
model.add(tf.layers.softmax());
model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: 'accuracy'
});

document.getElementById('start-training').addEventListener("click", () => {
  addStatus('\nLoading data...');
  document.getElementById('start-training').style.display = 'none';

  model.fitDataset(flattenedDataset, {
   epochs: 15,
   callbacks: {
     onEpochBegin: async (epoch) => {
       if (epoch === 0) {
         addStatus('Loaded data.');
         addStatus('Starting training.');
       }
       addStatus(`Start of Epoch ${epoch} / 14`);
     },
     onBatchEnd: async (batch, logs) => {
       addStatus(`Batch ${batch}: ${JSON.stringify(logs)}`);
     },
     onEpochEnd: async (epoch, logs) => {
       addStatus(`End of Epoch ${epoch} / 14: ${JSON.stringify(logs)}`);
     },
     onTrainEnd: async (logs) => {
       addStatus('Finished training. Downloading model files.');
       model.save('downloads://mnist_dnn');
     }
   }
  });
});

/************************************************************************
* Useful snippets
************************************************************************/

function addStatus(msg) {
  document.getElementById('status').innerText += '\n' + msg;
}

// Predict model and generate random tensor
// for (let i = 0; i < 10; i++) {
//   let x = tf.randomNormal([1, 784]);
//   let y_pred = model.predict(x);
//   text.innerText += y_pred.arraySync() + '\n';
// }

// Force CSVDataset to download (rather than lazy waiting until train time)
// csvDataset.columnNames().then(x => {console.log(x)});

// Extract one batch from the Dataset for debugging
// flattenedDataset.take(1).forEachAsync(e => { window.xs = e.xs.arraySync(); window.ys = e.ys.arraySync(); });
