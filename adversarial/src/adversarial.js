/**
 * Fast Gradient Sign Method (FGSM)
 *
 * This is an L_infinity attack (every pixel can change up to a maximum amount).
 *
 * Sources:
 * - [Goodfellow 15] Explaining and harnessing adversarial examples
 *
 * @param {tf.LayersModel} model - The model to construct an adversarial example for.
 * @param {tf.Tensor} img - The input image to construct an adversarial example for.
 * @param {tf.Tensor} lbl - The correct label of the image (must have shape [1, NUM_CLASSES]).
 * @param {Object} config - Optional configuration for this attack.
 * @param {number} config.ε - Max L_inf distance (each pixel can change up to this amount).
 *
 * @returns {tf.Tensor} The adversarial image.
 */
export function fgsm(model, img, lbl, {ε = 0.1} = {}) {
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

/**
 * Targeted Variant of the Fast Gradient Sign Method (FGSM)
 *
 * This is an L_infinity attack (every pixel can change up to a maximum amount).
 *
 * Sources:
 *  - [Kurakin 16] Adversarial examples in the physical world (original paper)
 *  - [Kurakin 16] Adversarial Machine Learning at Scale (best description)
 *
 * @param {tf.LayersModel} model - The model to construct an adversarial example for.
 * @param {tf.Tensor} img - The input image to construct an adversarial example for.
 * @param {tf.Tensor} lbl - The correct label of the image (must have shape [1, NUM_CLASSES]).
 * @param {tf.Tensor} targetLbl - The desired adversarial label of the image (must have shape [1, NUM_CLASSES]).
 * @param {Object} config - Optional configuration for this attack.
 * @param {number} config.ε - Max L_inf distance (each pixel can change up to this amount).
 * @param {number} config.loss - The loss function to use (must be 0, 1, or 2).
 *
 * @returns {tf.Tensor} The adversarial image.
 */
export function fgsmTargeted(model, img, lbl, targetLbl, {ε = 0.1, loss = 2} = {}) {
  // Loss functions (of increasing complexity) that measure how close the image is to the target class
  function loss0(input) {
    return tf.metrics.categoricalCrossentropy(targetLbl, model.predict(input));  // Make input closer to target class
  }
  function loss1(input) {
    return loss0(input).sub(tf.metrics.categoricalCrossentropy(lbl, model.predict(input)));  // + Move input away from original class
  }
  function loss2(input) {
    let NUM_CLASSES = lbl.shape[1];  // lbl is a one-hot vector in a batch of size 1
    let l = loss0(input).mul(5);
    for (let i = 0; i < NUM_CLASSES; i++) {
      if (i != tf.argMax(lbl)) {
        let nonlbl = tf.oneHot(i, NUM_CLASSES).reshape([1, NUM_CLASSES]);
        l = l.sub(tf.metrics.categoricalCrossentropy(nonlbl, model.predict(input)))  // + Move input away from all other classes
      }
    }
    return l;
  }
  let lossFn = [loss0, loss1, loss2][loss];

  // Perturb the image for one step in the direction of DECREASING loss
  let grad = tf.grad(lossFn);
  let delta = tf.sign(grad(img)).mul(ε);
  img = img.sub(delta).clipByValue(0, 1);

  return img;
}

/**
 * Basic Iterative Method (BIM / I-FGSM / PGD)
 *
 * This is an L_infinity attack (every pixel can change up to a maximum amount).
 *
 * Sources:
 *  - BIM: [Kurakin 16] Adversarial examples in the physical world
 *  - I-FGSM: [Tramer 17] Ensemble Adversarial Training: Attacks and Defenses
 *  - PGD: [Madry 19] Towards Deep Learning Models Resistant to Adversarial Attacks
 *
 * @param {tf.LayersModel} model - The model to construct an adversarial example for.
 * @param {tf.Tensor} img - The input image to construct an adversarial example for.
 * @param {tf.Tensor} lbl - The correct label of the image (must have shape [1, NUM_CLASSES]).
 * @param {Object} config - Optional configuration for this attack.
 * @param {number} config.ε - Max L_inf distance (each pixel can change up to this amount).
 * @param {number} config.α - Learning rate for gradient descent.
 * @param {number} config.iters - Number of iterations of gradient descent.
 *
 * @returns {tf.Tensor} The adversarial image.
 */
export function bim(model, img, lbl, {ε = 0.1, α = 0.01, iters = 10} = {}) {
  // Loss function that measures how close the image is to the original class
  function loss(input) {
    return tf.metrics.categoricalCrossentropy(lbl, model.predict(input));  // Make input farther from original class
  }

  // Random initialization for the PGD (for even better performance, we should try multiple inits)
  let aimg = img.add(tf.randomUniform(img.shape, -ε, ε)).clipByValue(0, 1);

  // Run PGD to MAXIMIZE the loss w.r.t. aimg
  let grad = tf.grad(loss);
  for (let i = 0; i < iters; i++) {
    let delta = tf.sign(grad(aimg)).mul(α);
    aimg = aimg.add(delta);
    aimg = tf.minimum(1, tf.minimum(img.add(ε), tf.maximum(0, tf.maximum(img.sub(ε), aimg))));  // Clips aimg to ε distance of img
  }

  return aimg;
}

/**
 * Targeted Variant of the Basic Iterative Method (BIM / I-FGSM / PGD)
 *
 * This is an L_infinity attack (every pixel can change up to a maximum amount).
 *
 * Sources:
 *  - [Kurakin 16] Adversarial examples in the physical world (original paper)
 *  - [Kurakin 16] Adversarial Machine Learning at Scale (best description)
 *
 * @param {tf.LayersModel} model - The model to construct an adversarial example for.
 * @param {tf.Tensor} img - The input image to construct an adversarial example for.
 * @param {tf.Tensor} lbl - The correct label of the image (must have shape [1, NUM_CLASSES]).
 * @param {tf.Tensor} targetLbl - The desired adversarial label of the image (must have shape [1, NUM_CLASSES]).
 * @param {Object} config - Optional configuration for this attack.
 * @param {number} config.ε - Max L_inf distance (each pixel can change up to this amount).
 * @param {number} config.iters - Number of iterations of gradient descent.
 * @param {number} config.loss - The loss function to use (must be 0 or 1). Note: loss2 from fgsmTargeted theoretically works, but it's too slow in practice.
 *
 * @returns {tf.Tensor} The adversarial image.
 */
export function bimTargeted(model, img, lbl, targetLbl, {ε = 0.1, α = 0.01, iters = 10, loss = 1} = {}) {
  // Loss functions (of increasing complexity) that measure how close the image is to the target class
  function loss0(input) {
    return tf.metrics.categoricalCrossentropy(targetLbl, model.predict(input));  // Make input closer to target class
  }
  function loss1(input) {
    return loss0(input).sub(tf.metrics.categoricalCrossentropy(lbl, model.predict(input)));  // + Move input away from original class
  }
  let lossFn = [loss0, loss1][loss];

  // Random initialization for the PGD (for even better performance, we should try multiple inits)
  let aimg = img.add(tf.randomUniform(img.shape, -ε, ε)).clipByValue(0, 1);

  // Run PGD to MINIMIZE the loss w.r.t. aimg
  let grad = tf.grad(lossFn);
  for (let i = 0; i < iters; i++) {
    let delta = tf.sign(grad(aimg)).mul(α);
    aimg = aimg.sub(delta);
    aimg = tf.minimum(1, tf.minimum(img.add(ε), tf.maximum(0, tf.maximum(img.sub(ε), aimg))));  // Clips aimg to ε distance of img
  }

  return aimg;
}

/**
 * One-Pixel Variant of the Jacobian-based Saliency Map Attack (JSMA / JSMA-F)
 *
 * This is an L0 attack (we can change a limited number of pixels as much as we want).
 *
 * This is a much simplified version of the normal JSMA attack, where we only
 * consider single pixels at a time, rather than pairs of pixels. Additionally,
 * instead of computing the full saliency, we rely only on the gradient of the
 * target class wrt the image. This is much faster and scalable than JSMA, and
 * has similar performance on MNIST and CIFAR-10.
 *
 * Sources:
 *  - JSMA: [Papernot 15] The Limitations of Deep Learning in Adversarial Settings
 *  - JSMA-F: [Carlini 17] Towards Evaluating the Robustness of Neural Networks
 *
 * @param {tf.LayersModel} model - The model to construct an adversarial example for.
 * @param {tf.Tensor} img - The input image to construct an adversarial example for.
 * @param {tf.Tensor} lbl - The correct label of the image (must have shape [1, NUM_CLASSES]).
 * @param {Object} config - Optional configuration for this attack.
 * @param {number} config.ε - Max L0 distance (we can change up to this many pixels).
 *
 * @returns {tf.Tensor} The adversarial image.
 */
export function jsmaOnePixel(model, img, lbl, targetLbl, {ε = 28} = {}) {
  // Compute useful constants
  let NUM_PIXELS = img.flatten().shape[0];  // Number of pixels in the image (for RGB, each channel counts as one "pixel")

  // Function that outputs the target class probability of an image (used for per-pixel saliency)
  function classProb(img) {
    return tf.dot(model.predict(img), targetLbl.squeeze());
  }

  // We copy image data into an array to easily make per-pixel perturbations
  let imgArr = img.flatten().arraySync();

  // Track what pixels we've changed and should not change again (set bit to 0 in this mask)
  let changedPixels = tf.ones([NUM_PIXELS]).arraySync();

  // Modify the pixel with the highest impact (gradient) on the target class probability, and repeat
  let grad = tf.grad(classProb);
  let aimg = tf.tensor(imgArr, img.shape);
  for (let i = 0; i < ε; i++) {
    // Compute highest impact pixel to change, and update that pixel in imgArr
    tf.tidy(() => {  // (This should be in tf.tidy() since the number of iterations can be large for ImageNet)
      // Compute pixel with maximum gradient value
      // (Note: in this simplified variant, we just use the raw gradient as the "saliency",
      //  not the more robust saliency function in the paper (Eq. 8).)
      let g = grad(aimg);
      let changedPixelsMask = tf.tensor(changedPixels);
      let pixelToChange = tf.mul(g.flatten(), changedPixelsMask).argMax().dataSync()[0];

      // Change that pixel in the image data array
      imgArr[pixelToChange] = 1;

      // Remember that we've modified this pixel
      changedPixels[pixelToChange] = 0;
    });

    // Update the aimg tensor with the latest imgArr data
    aimg.dispose();  // Delete old data, otherwise the old tensor becomes an orphaned memory leak
    aimg = tf.tensor(imgArr, img.shape);
  }

  return aimg;
}

/**
 * Jacobian-based Saliency Map Attack (JSMA / JSMA-F)
 *
 * This is an L0 attack (we can change a limited number of pixels as much as we want).
 *
 * (Note: I tried JSMA-Z as well, which uses logits instead of softmax probabilities.
 *  This results in much worse performance for this attack, even though JSMA-Z was
 *  the original variant of this attack (see Carlini 17). I'm not sure why there's
 *  a huge discrepancy.)
 *
 * Sources:
 *  - JSMA: [Papernot 15] The Limitations of Deep Learning in Adversarial Settings
 *  - JSMA-F: [Carlini 17] Towards Evaluating the Robustness of Neural Networks
 *
 * @param {tf.LayersModel} model - The model to construct an adversarial example for.
 * @param {tf.Tensor} img - The input image to construct an adversarial example for.
 * @param {tf.Tensor} lbl - The correct label of the image (must have shape [1, NUM_CLASSES]).
 * @param {tf.Tensor} targetLbl - The desired adversarial label of the image (must have shape [1, NUM_CLASSES]).
 * @param {Object} config - Optional configuration for this attack.
 * @param {number} config.ε - Max L0 distance (we can change up to this many pixels).
 *
 * @returns {tf.Tensor} The adversarial image.
 */
export function jsma(model, img, lbl, targetLbl, {ε = 28} = {}) {
  // Compute useful constants
  let NUM_PIXELS = img.flatten().shape[0];      // Number of pixels in the image (for RGB, each channel counts as one "pixel")
  let LT = targetLbl.argMax(1).arraySync()[0];  // Target label as an index rather than a one-hot vector
  if (NUM_PIXELS > 32*32*3) { throw 'JSMA does not scale to images larger than CIFAR-10 (32x32x3)!'; }

  // Function that outputs the target class probability of an image (used for per-pixel saliency)
  let classProbs = [];
  for (let l = 0; l < 10; l++) {
    classProbs.push(img => tf.dot(model.predict(img), tf.oneHot(l, 10)));
  }

  // We copy image data into an array to easily make per-pixel perturbations
  let imgArr = img.flatten().arraySync();

  // Track what pixels we've changed and should not change again (set bit to 0 in this mask)
  let changedPixels = tf.ones([NUM_PIXELS, NUM_PIXELS]).arraySync();
  for (let p = 0; p < NUM_PIXELS; p++) {
    changedPixels[p][p] = 0;  // (p, p) is not a valid pair of two different pixels
  }

  // Modify the pixel pair with the highest impact (saliency) on the target class probability, and repeat
  let grads = classProbs.map(classProb => tf.grad(classProb));
  let aimg = tf.tensor(imgArr, img.shape);
  for (let i = 0; i < Math.floor(ε/2); i++) {
    // Compute highest impact pixel pair to change, and update that pixel pair in imgArr
    tf.tidy(() => {  // (This must be in tf.tidy() as there are many intermediate NUM_PIXELS^2 matrices)
      // Compute all gradients ∂classProb / ∂img
      let gs = [];
      for (let l = 0; l < 10; l++) { gs.push(grads[l](aimg)); }

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

      // Change that pixel pair in the image data array
      imgArr[p] = 1;
      imgArr[q] = 1;

      // Remember that we've modified this pixel pair
      for (let j = 0; j < NUM_PIXELS; j++) {
        changedPixels[j][p] = 0;
        changedPixels[j][q] = 0;
        changedPixels[p][j] = 0;
        changedPixels[q][j] = 0;
      }
    });

    // Update the aimg tensor with the latest imgArr data
    aimg.dispose();  // Delete old data, otherwise the old tensor becomes an orphaned memory leak
    aimg = tf.tensor(imgArr, img.shape);
  }

  return aimg;
}

/**
 * Carlini & Wagner (C&W)
 *
 * This is an L2 attack (we are incentivized to change many pixels by very small amounts).
 *
 * Note that this attack does NOT allow us to set a maximum L2 perturbation.
 *
 * Sources:
 *  - [Carlini 17] Towards Evaluating the Robustness of Neural Networks
 *  - [Carlini 17] Adversarial Examples Are Not Easily Detected - Bypassing Ten Detection Methods
 *
 * @param {tf.LayersModel} model - The model to construct an adversarial example for.
 * @param {tf.Tensor} img - The input image to construct an adversarial example for.
 * @param {tf.Tensor} lbl - The correct label of the image (must have shape [1, NUM_CLASSES]).
 * @param {tf.Tensor} targetLbl - The desired adversarial label of the image (must have shape [1, NUM_CLASSES]).
 * @param {Object} config - Optional configuration for this attack.
 * @param {number} config.c - Higher = higher success rate, but higher distortion.
 * @param {number} config.κ - Higher = more confident adv example.
 * @param {number} config.λ - Higher learning rate = faster convergence, but higher distortion.
 * @param {number} config.iters - Number of iterations of gradient descent (Adam).
 *
 * @returns {tf.Tensor} The adversarial image.
 */
export function cw(model, img, lbl, targetLbl, {c = 5, κ = 1, λ = 0.1, iters = 100} = {}) {
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

/**
* Returns the [row, col] coordinate of the maximum element in a square matrix.
*/
function argmax2d(m) {
  if (m.shape[0] !== m.shape[1]) { throw 'argmax2d() only supports square matrices!'; }
  let N = m.shape[0];
  let idx = tf.argMax(m.reshape([-1])).dataSync()[0];
  let row = Math.floor(idx / N);
  let col = idx % N;
  return [row, col];
}

/**
* Sanity test of the argmax2d() utility function.
*/
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
  // Our monkey-patched MobileNet model has a predictLogits method already
  if (model.predictLogits !== undefined) { return {predict: img => model.predictLogits(img)}; }

  // Otherwise, for LayersModels, clone the model with all layers except the last one
  if (!model.layers[model.layers.length-1].name.includes('softmax')) { throw 'The last layer of the model must be softmax!' }
  let logitsModel = tf.sequential();
  for (let i = 0; i < model.layers.length-1; i++) {
    logitsModel.add(model.layers[i]);
  }
  return logitsModel;
}