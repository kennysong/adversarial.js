<template>
  <div class="dropdown">
    <button class="dropdown-button dropdown-toggle" type="button" id="select-model" data-bs-toggle="dropdown" aria-expanded="false" value = "">
      {{key}}
    </button>
      <ul class="dropdown-menu" aria-labelledby="select-model">
        <li v-for="option in options" :key="option.key">
          <a class="dropdown-item" @click="onClick(option.value); key = option.key; value = option.value" href="javascript: void(0)">{{option.key}}</a>
        </li>
      </ul>
  </div>
</template>
  
<script>
import {changeArchitecture} from "../../public/js/intro.js"
import "bootstrap/dist/css/bootstrap.min.css"
import "bootstrap"
export default {
  name: 'ModelDropDown',
  id: 'select-model',
  props: {
    description: String
  },
  data: () => ({
    options: [
      { key: "VGG-16", value: 'vgg16' }, 
      { key: "ResNet", value: 'resnet' }, 
      { key: "Xception", value: 'xception'},
      { key: "MobileNet", value: 'mobilenet' }
      ],
    key: "Select Architecture",
    value: ""
  }),
  methods: {
	onClick(value){
		changeArchitecture(value)
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
text-align: left;
}
</style>