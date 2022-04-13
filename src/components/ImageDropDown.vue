<template>
  <div class="dropdown">
    <button class="dropdown-button dropdown-toggle" type="button" id="select-target" data-bs-toggle="dropdown" aria-expanded="false" value = "">
      {{key}}
    </button>
      <ul class="dropdown-menu" aria-labelledby="select-target">
        <li v-for="option in options" :key="option.key">
          <a class="dropdown-item" @click="onClick(option.value); key = option.key; value = option.value" href="javascript: void(0)">{{option.key}}</a>
        </li>
      </ul>
  </div>
</template>
  
<script>
import {changeTarget} from "../../public/js/intro.js"
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
      { key: "Potato", value: '934' },
      { key: "Street Lamp", value: '413' }, 
      { key: "Sushi", value: '151' }, 
      { key: "Donkey", value: '24'}
      ],
    key: "Select Target",
    value: "0"
  }),
  methods: {
	onClick(value){
		changeTarget(value)
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
</style>
