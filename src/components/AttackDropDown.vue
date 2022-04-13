<template>
  <div class="dropdown">
        <button class="dropdown-button dropdown-toggle" type="button" id="select-attack" data-bs-toggle="dropdown" aria-expanded="false" value="">
      {{key}}
    </button>
      <ul class="dropdown-menu" aria-labelledby="select-attack">
        <li v-for="option in options" :key="option.key">
          <a class="dropdown-item" @click="onClick(option.value); key = option.key; value = option.value" href="javascript: void(0)">{{option.key}}</a>
        </li>
      </ul>
  </div>
</template>

<script>
import {changeAttack} from "../../public/js/intro.js"
import "bootstrap/dist/css/bootstrap.min.css"
import "bootstrap"
export default {
  name: 'AttackDropDown',
  id: 'select-attack',
  props: {
    description: String
  },
  data: () => ({
    options: [
      { key: "Carlini & Wagner (strongest) (digit recognition)", value: 'cw' },
      { key: "Jacobian-based Saliency Map Attack", value: 'jsma' },
      { key: "Jacobian-based Saliency Map Attack 1-Pixel (stronger)", value: 'jsmaOnePixel' },
      { key: "Basic Iterative Method (stronger)", value: 'bimTargeted' }, 
      { key: "Fast Gradient Sign Method (weak)", value: 'imagenet'}
      ],
	key: "Select Attack",
	value: ""
  }),
  methods: {
	onClick(value){
		changeAttack(value)
	}
  }
}
</script>

<style>
.dropdown-button {
background-color: #FFFFFF;
color: #000000;
font-family: "Raleway";
font-size: 15px;
width: 24em;
height: 4em;
box-shadow: 1px 2px 4px #cecece;
border-radius: .3em;
border: 0;
text-align: left;
}

.dropdown-toggle::after {
  float: right !important;
  vertical-align: middle !important;
}
</style>