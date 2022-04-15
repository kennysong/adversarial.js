<template>
    <div class="main-container">
      <Navigation pageTitle="Home"/>
      <Tutorial/>
      <Container/>
      <Footer/>
    </div>
</template>

<script>
import Navigation from './components/Navigation.vue'
import Tutorial from './components/Tutorial.vue'
import Container from './components/Container.vue'
import Footer from './components/Footer.vue'
import "bootstrap/dist/css/bootstrap.min.css"
import "bootstrap"

export default {
  name: 'App',
  components: {
    Navigation,
    Container,
    Footer,
    Tutorial
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
    background-color: #E0E8F3;
  }

  .main-container {
    width: 100%;
    margin: 0 auto;
  }
</style>
