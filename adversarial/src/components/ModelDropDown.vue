<template>
  <div class="dropdown">
      <button class="btn btn-primary dropdown-toggle" type="button" id="select-model" data-bs-toggle="dropdown" aria-expanded="false" value = "">
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
      { key: "LeNet", value: 'lenet' },
      { key: "VGG-16", value: 'vgg16' }, 
      { key: "ResNet", value: 'resnet' }, 
      { key: "MobileNet", value: 'mobilenet'}
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
</style>