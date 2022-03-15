<template>
  <div class="row center" style="display:flex;"> <!-- The flex here keeps the child columns the same height -->
          <div class="one-half column border-right">
            <div id="original-prediction-overlay" class="overlay"></div>
            <h5>Prediction</h5>
            <Button value = "Run Neural Network" id = "predict-original"/>
            <div id="prediction"></div>
            <div id="prediction-status"></div>
          </div>
          <div class="one-half column">
            <div id="adversarial-prediction-overlay" class="overlay"></div>
            <h5>Prediction</h5>
            <button id="predict-adv" class="button-primary" disabled>Click "Generate" First</button>
            <div id="prediction-adv"></div>
            <div id="prediction-adv-status"></div>
          </div>
        </div>
</template>

<script>
import Button from './Button.vue'

export default {
  name: 'Footer',
  props: {
    description: String
  },
  components: {
    Button
  }
}
</script>

<style>

</style>