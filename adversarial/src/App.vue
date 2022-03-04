<template>
    <div class="main-container">
      <Navigation pageTitle="Home"/>
      <Button description="description" />
      <DropDown description="description" />
      <Footer description="description" />
      <Image description="description" />
      <FAQ description="description" />
    </div>
</template>

<script>
import Navigation from './components/Navigation.vue'
import Button from './components/Button.vue'
import DropDown from './components/DropDown.vue'
import Footer from './components/Footer.vue'
import Image from './components/Image.vue'
import FAQ from './components/FAQ.vue'

export default {
  name: 'App',
  components: {
    Navigation,
    Button,
    DropDown,
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
