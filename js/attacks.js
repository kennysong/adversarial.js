/************************************************************************
* FGSM
*  - [Goodfellow 15] Explaining and harnessing adversarial examples
************************************************************************/

export function fgsm(model, img, lbl) {
  let ε = 0.2;  // L_inf distance (each pixel can change up to this amount)
  
  // Loss function that measures how close the image is to the original class
  function loss(input) {
    return tf.metrics.categoricalCrossentropy(lbl, model.predict(input));  // Make input farther from original class
  }
  
  // Perturb the image for one step in the direction of INCREASING loss
  let grad = tf.grad(loss);
  let delta = tf.sign(grad(img)).mul(ε);
  img = img.add(delta).clipByValue(0, 1);

  return img;
}

/************************************************************************
* FGSM (targeted variant)
*  - [Kurakin 16] Adversarial Machine Learning at Scale (best description)
*  - [Kurakin 16] Adversarial examples in the physical world (first reference)
************************************************************************/

export function fgsmTargeted(model, img, lbl, targetLbl) {
  let ε = 0.2;  // L_inf distance (each pixel can change up to this amount)
  let α = 0.01;
  
  // Loss functions (of increasing complexity) that measure how close the image is to the target class
  function loss(input) {
    return tf.metrics.categoricalCrossentropy(targetLbl, model.predict(input));  // Make input closer to target class
  }
  function loss2(input) {
    return loss(input).sub(tf.metrics.categoricalCrossentropy(lbl, model.predict(input)));  // + Move input away from original class
  }
  function loss3(input) {
    let l = loss(input).mul(5);
    for (let i = 0; i < 10; i++) {
      if (i != tf.argMax(lbl)) {
        let nonlbl = tf.oneHot(i, 10).reshape([1, 10]);
        l = l.sub(tf.metrics.categoricalCrossentropy(nonlbl, model.predict(input)))  // + Move input away from all other classes
      }
    }
    return l;
  }
  
  // Perturb the image for one step in the direction of DECREASING loss
  let grad = tf.grad(loss3);
  let delta = tf.sign(grad(img)).mul(ε);
  img = img.sub(delta).clipByValue(0, 1);

  return img;
}

/************************************************************************
* BIM / I-FGSM / PGD
*  - BIM: [Kurakin 16] Adversarial examples in the physical world
*  - I-FGSM: [Tramer 17] Ensemble Adversarial Training: Attacks and Defenses
*  - PGD: [Madry 19] Towards Deep Learning Models Resistant to Adversarial Attacks
************************************************************************/

export function bim(model, img, lbl) {
  let ε = 0.2;     // L_inf distance (all pixels can change up to this amount)
  let α = 0.01;    // Learning rate for PGD
  let iters = 50;  // Number of iterations of PGD
  
  // Loss function that measures how close the image is to the original class
  function loss(input) {
    return tf.metrics.categoricalCrossentropy(lbl, model.predict(input));  // Make input farther from original class
  }
  
  // Random initialization for the PGD (for even better performance, we should try multiple inits)
  img = img.add(tf.randomUniform(img.shape, -ε, ε));
  
  // Run PGD to MAXIMIZE the loss w.r.t. img
  let grad = tf.grad(loss);
  for (let i = 0; i < iters; i++) {
    let delta = tf.sign(grad(img)).mul(α);
    img = img.add(delta);
    img = tf.minimum(1, tf.minimum(img.add(ε), tf.maximum(0, tf.maximum(img.sub(ε), img))));  // Clips img to ε distance of img
  }

  return img;
}

/************************************************************************
* BIM / I-FGSM / PGD (targeted variant)
*  - [Kurakin 16] Adversarial Machine Learning at Scale (best description)
*  - [Kurakin 16] Adversarial examples in the physical world (first reference)
************************************************************************/

export function bimTargeted(model, img, lbl, targetLbl) {
  let ε = 0.2;     // L_inf distance (all pixels can change up to this amount)
  let α = 0.01;    // Learning rate for PGD
  let iters = 50;  // Number of iterations of PGD
  
  // Loss functions (of increasing complexity) that measure how close the image is to the target class
  function loss(input) {
    return tf.metrics.categoricalCrossentropy(targetLbl, model.predict(input));  // Make input closer to target class
  }
  function loss2(input) {
    return loss(input).sub(tf.metrics.categoricalCrossentropy(lbl, model.predict(input)));  // + Move input away from original class
  }
  function loss3(input) {
    let l = loss(input).mul(5);
    for (let i = 0; i < 10; i++) {
      if (i != tf.argMax(lbl)) {
        let nonlbl = tf.oneHot(i, 10).reshape([1, 10]);
        l = l.sub(tf.metrics.categoricalCrossentropy(nonlbl, model.predict(input)))  // + Move input away from all other classes
      }
    }
    return l;
  }
  
  // Random initialization for the PGD (for even better performance, we should try multiple inits)
  img = img.add(tf.randomUniform(img.shape, -ε, ε));
  
  // Run PGD to MINIMIZE the loss w.r.t. img
  let grad = tf.grad(loss2); // Loss3 is too slow
  for (let i = 0; i < iters; i++) {
    let delta = tf.sign(grad(img)).mul(α);
    img = img.sub(delta);
    img = tf.minimum(1, tf.minimum(img.add(ε), tf.maximum(0, tf.maximum(img.sub(ε), img))));  // Clips img to ε distance of img
  }

  return img;
}

/************************************************************************
* JSMA / JSMA-F (one pixel variant)
*  - JSMA: [Papernot 15] The Limitations of Deep Learning in Adversarial Settings
*  - JSMA-F: [Carlini 17] Towards Evaluating the Robustness of Neural Networks
************************************************************************/

export function jsmaOnePixel(model, img, lbl, targetLbl) {
  let ε = 30;  // L0 distance (we can change up to this many pixels)
  
  // Compute useful constants
  let NUM_PIXELS = img.flatten().shape[0];  // Number of pixels in the image (for RGB, each channel counts as one "pixel")
  
  // Function that outputs the target class probability of an image (used for per-pixel saliency)
  function classProb(img) {
    return tf.dot(model.predict(img), targetLbl.squeeze());
  }
  
  // Track what pixels we've changed and should not change again (set bit to 0 in this mask)
  let changedPixels = tf.ones([NUM_PIXELS]).arraySync();
  
  // Modify the pixel with the highest impact (gradient) on the target class probability, and repeat
  let grad = tf.grad(classProb);
  for (let i = 0; i < ε; i++) {
    // Compute pixel with maximum gradient value
    // (Note: in this simplified variant, we just use the raw gradient as the "saliency", 
    //  not the more robust saliency function in the paper (Eq. 8).)
    let g = grad(img);
    let changedPixelsMask = tf.tensor(changedPixels);
    let pixelToChange = tf.mul(g.flatten(), changedPixelsMask).argMax().dataSync()[0];
    
    // Modify that pixel in the image
    let imgArr = img.flatten().arraySync();
    imgArr[pixelToChange] = 1;
    img = tf.tensor(imgArr, img.shape);
        
    // Remember that we've modified this pixel
    changedPixels[pixelToChange] = 0;
  }
  
  return img;
}

/************************************************************************
* JSMA / JSMA-F
*  - JSMA: [Papernot 15] The Limitations of Deep Learning in Adversarial Settings
*  - JSMA-F: [Carlini 17] Towards Evaluating the Robustness of Neural Networks
*  (Note: For some reason, using logits instead of softmax probabilities results in much worse
*   performance for this attack. I'm not sure why there's a huge discrepancy.)
************************************************************************/

export function jsma(model, img, lbl, targetLbl) {
  let ε = 30;  // L0 distance (we can change up to this many pixels)
  
  // Compute useful constants
  let NUM_PIXELS = img.flatten().shape[0];      // Number of pixels in the image (for RGB, each channel counts as one "pixel")
  let LT = targetLbl.argMax(1).arraySync()[0];  // Target label as an index rather than a one-hot vector

  // Function that outputs the target class probability of an image (used for per-pixel saliency)
  let classProbs = [];
  for (let l = 0; l < 10; l++) {
    classProbs.push(img => tf.dot(model.predict(img), tf.oneHot(l, 10)));
  }
  
  // Track what pixels we've changed and should not change again (set bit to 0 in this mask)
  let changedPixels = tf.ones([NUM_PIXELS, NUM_PIXELS]).arraySync();
  for (let p = 0; p < NUM_PIXELS; p++) { 
    changedPixels[p][p] = 0;  // (p, p) is not a valid pair of two different pixels
  }
  
  // Modify the pixel pair with the highest impact (saliency) on the target class probability, and repeat
  let grads = classProbs.map(classProb => tf.grad(classProb));
  for (let i = 0; i < Math.floor(ε/2); i++) {
    tf.tidy(() => {  // (This must be in tf.tidy() as there are many intermediate NUM_PIXELS^2 matrices)
      // Compute all gradients ∂classProb / ∂img
      let gs = [];
      for (let l = 0; l < 10; l++) { gs.push(grads[l](img)); }

      // Compute α_pq for all pairs of pixels (p, q), vectorized using an outer sum
      // (Outer sum works by broadcasting: https://stackoverflow.com/a/33848814/908744)
      let α = tf.add(gs[LT].reshape([1, NUM_PIXELS]), gs[LT].reshape([NUM_PIXELS, 1]))

      // Compute β_pq for all pairs of pixels (p, q)
      // (Note that we swap the order of summations from the paper pseudocode. In case it's 
      // not obvious we can do that, see: https://math.stackexchange.com/a/1931615/28855)
      let β = tf.zerosLike(α);
      for (let l = 0; l < 10; l++) {
        if (l === LT) { continue; }
        β = β.add(tf.add(gs[l].reshape([1, NUM_PIXELS]), gs[l].reshape([NUM_PIXELS, 1])))
      }

      // Compute the best (highest saliency) pair of pixels (p, q)
      let saliencyGrid = tf.mul(α.neg(), β).mul(α.greater(0)).mul(β.less(0));
      let changedPixelsMask = tf.tensor(changedPixels);
      let [p, q] = argmax2d(saliencyGrid.mul(changedPixelsMask));

      // Modify that pixel in the image
      let imgArr = img.flatten().arraySync();
      imgArr[p] = 1;
      imgArr[q] = 1;
      img = tf.tensor(imgArr, img.shape);

      // Remember that we've modified this pixel
      for (let j = 0; j < NUM_PIXELS; j++) { 
        changedPixels[j][p] = 0; 
        changedPixels[j][q] = 0;
        changedPixels[p][j] = 0;
        changedPixels[q][j] = 0;
      }
      
      // Make sure our global tensors are not garbage collected by tf.tidy()
      img = tf.keep(img);
      changedPixels = tf.keep(changedPixels);
    });
  }
  
  return img;
}

/************************************************************************
* C&W
*  - [Carlini 17] Towards Evaluating the Robustness of Neural Networks
*  - [Carlini 17] Adversarial Examples Are Not Easily Detected - Bypassing Ten Detection Methods
*  (The description in the latter paper is easier to understand)
************************************************************************/

export function cw(model, img, lbl, targetLbl) {
  let c = 5;        // Higher = higher success rate, but higher distortion
  let κ = 0.2;      // Higher = more confident adv example
  let λ = 0.1;      // Higher learning rate = faster convergence, but higher distortion
  let iters = 250;  // Iterations of gradient descent to produce adv example
                    // Note: This attack does not allow us to set a max L2 distance!
  
  // C&W requires using logits, rather than softmax class probabilities
  let modelLogits = getModelLogits(model);
  
  // The tf.train.Optimizer API mutates this global tf.Variable
  // (What is w? "Instead of δ, we apply a change-of-variables and optimize over w.")
  let w = tf.variable(tf.atanh(tf.sub(tf.mul(img, 2), 1)));

  // The C&W attack's objective function (technically takes an argument w, but we access the global w)
  function cwObjective() {
    // We map w back into an image and measure the L2 distortion from the original image
    let wImg = tf.mul(tf.add(tf.tanh(w), 1), 0.5);
    let distortion = tf.square(tf.norm(tf.sub(img, wImg), 2));
    
    // Compute the "not predicting desired class yet" part of the objective
    let logits = modelLogits.predict(wImg);
    let targetLblLogit = tf.dot(logits, targetLbl.squeeze());
    let otherLblMaxLogit = tf.max(tf.sub(tf.mul(tf.sub(1, targetLbl), logits), tf.mul(targetLbl, 10000)));  // Copied from: https://github.com/carlini/nn_robust_attacks/blob/610c43f9e3c8def7b44b6909a79aceac073d2fba/l2_attack.py#L97
    let loss = tf.maximum(tf.sub(otherLblMaxLogit, targetLblLogit), -κ);
    
    return tf.add(distortion, tf.mul(c, loss)).squeeze();  // squeeze() because the tf.train.Optimizer API requires we return a scalar
  }

  // Run Adam on w to MINIMIZE the C&W objective
  let opt = tf.train.adam(0.1);
  for (let i = 0; i < iters; i++) {
    opt.minimize(cwObjective, false, [w]);
  }
  
  // Map w back into an image and return it
  return tf.mul(tf.add(tf.tanh(w), 1), 0.5);
}

/************************************************************************
* Utils
************************************************************************/

function argmax2d(m) {
  if (m.shape[0] !== m.shape[1]) { throw 'argmax2d() only supports square matrices!'; }
  let N = m.shape[0];
  let idx = tf.argMax(m.reshape([-1])).dataSync()[0];
  let row = Math.floor(idx / N);
  let col = idx % N;
  return [row, col];
}

function testArgmax2d() {
  for (let i = 0; i < 10; i++) {
    let m = tf.randomUniform([784, 784]);
    let [r, c] = argmax2d(m);
    console.assert(m.max().dataSync()[0] === m.arraySync()[r][c]);
  }
}

/** 
* Returns a copy of model without the softmax layer, so it predict()'s logits.
*/
function getModelLogits(model) {
  if (!model.layers[model.layers.length-1].name.includes('softmax')) { throw 'The last layer of the model must be softmax!' }
  let logitsModel = tf.sequential();
  for (let i = 0; i < model.layers.length-1; i++) {
    logitsModel.add(model.layers[i]);
  }
  return logitsModel;
}