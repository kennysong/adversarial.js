<template>
    <div class="main-container">
      <Navigation pageTitle="Home"/>
    </div>
</template>

<script>
import Navigation from './components/Navigation.vue'

export default {
  name: 'App',
  components: {
    Navigation
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
