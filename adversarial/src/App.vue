<template>
    <div class="main-container">
      <Navigation pageTitle="Home"/>
      <Button value = "Generate" id = "GenerateButton" />

      <Button value = "Prediction"/>
      <Button value = "Upload Image"/>
      <ModelDropDown/>
      <AttackDropDown/>
      <ImageDropDown/>
      <div class="row border-bottom">
          <div class="one-half column split border-right">
            <div id="original-canvas-overlay" class="overlay"></div>
            <div class="splitleft61">
              <canvas id="original" class="tableimg" width="224" height="224"></canvas>
            </div>
            <div class="splitright39">
              <Button value = "Next Image" id = "next-image"/>
            </div>
          </div>
        </div>

      <Footer/>
      <Image/>
      <FAQ/>
    </div>
</template>

<script>
import Navigation from './components/Navigation.vue'
import Button from './components/Button.vue'
import ModelDropDown from './components/ModelDropDown.vue'
import AttackDropDown from './components/AttackDropDown.vue'
import ImageDropDown from './components/ImageDropDown.vue'
import Footer from './components/Footer.vue'
import Image from './components/Image.vue'
import FAQ from './components/FAQ.vue'
import "bootstrap/dist/css/bootstrap.min.css"
import "bootstrap"

export default {
  name: 'App',
  components: {
    Navigation,
    Button,
    ModelDropDown,
    AttackDropDown,
    ImageDropDown,
    Footer,
    Image,
    FAQ
  }
}
</script>

<style>
  @font-face {
    font-family: "Raleway";
    src: url(./fonts/raleway/Raleway[wght].ttf) format("truetype");
  }
  @font-face {
    font-family: "Raleway-italic";
    src: url(./fonts/raleway/Raleway-Italic[wght].ttf) format("truetype");
  }
  @font-face {
    font-family: "Montserrat";
    src: url(./fonts/Montserrat/Montserrat-VariableFont_wght.ttf) format("truetype");
  }

  body {
    margin: 0;
  }

  .main-container {
    width: 100%;
    margin: 0 auto;
  }
</style>
