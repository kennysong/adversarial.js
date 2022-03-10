<template>
    <div class="main-container">
      <Navigation pageTitle="Home"/>
      <Button value = "Generate"/>
      <Button value = "Prediction"/>
      <Button value = "Upload Image"/>
      <DropDown Select Model/>
        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton1">
        <li><a class="dropdown-item" href="#">MNIST (digit recognition)</a></li>
        <li><a class="dropdown-item" href="#">GTSRB (street sign recognition)</a></li>
        <li><a class="dropdown-item" href="#">CIFAR-10 (object recognition, small)</a></li>
        <li><a class="dropdown-item" href="#">ImageNet (object recognition, large)</a></li>
        </ul>
      <div class="dropdown">
      <button class="btn btn-secondary dropdown-toggle" type="button" id="dropdownMenuButton1" data-bs-toggle="dropdown" aria-expanded="false">
      Select Model
      </button>
        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton1">
        <li><a class="dropdown-item" href="#">MNIST (digit recognition)</a></li>
        <li><a class="dropdown-item" href="#">GTSRB (street sign recognition)</a></li>
        <li><a class="dropdown-item" href="#">CIFAR-10 (object recognition, small)</a></li>
        <li><a class="dropdown-item" href="#">ImageNet (object recognition, large)</a></li>
        </ul>
      </div>
      <div class="dropdown">
      <button class="btn btn-primary dropdown-toggle" type="button" id="dropdownMenuButton1" data-bs-toggle="dropdown" aria-expanded="false">
      Select Attack
      </button>
        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton1">
        <li><a class="dropdown-item" href="#">Carlini & Wagner (strongest) (digit recognition)</a></li>
        <li><a class="dropdown-item" href="#">Jacobian-based Saliency Map Attack</a></li>
        <li><a class="dropdown-item" href="#">Basic Iterative Method (stronger)</a></li>
        <li><a class="dropdown-item" href="#">Fast Gradient Sign Method (weak)</a></li>
        </ul>
      </div>
      <div class="dropdown">
      <button class="btn btn-info dropdown-toggle" type="button" id="dropdownMenuButton1" data-bs-toggle="dropdown" aria-expanded="false">
      Select Image
      </button>
        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton1">
        <li><a class="dropdown-item" href="#">Potato</a></li>
        <li><a class="dropdown-item" href="#">Street Lamp</a></li>
        <li><a class="dropdown-item" href="#">Sushi</a></li>
        <li><a class="dropdown-item" href="#">Donkey </a></li>
        </ul>
      </div>
      
      <Footer/>
      <Image/>
      <FAQ/>
    </div>
</template>

<script>
import Navigation from './components/Navigation.vue'
import Button from './components/Button.vue'
import DropDown from './components/DropDown.vue'
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
