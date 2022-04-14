<template>
  <div v-if="imageDropdown">
    <button class="dropdown-button dropdown-toggle" type="button" id="select-target" data-bs-toggle="dropdown" aria-expanded="false" value = "">
      {{key}}
    </button>
      <ul class="dropdown-menu" aria-labelledby="select-target">
        <li v-for="option in imageDropdown.options" :key="option.key">
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
    description: String,
    newDataset: String,
  },
  data: () => ({
    datasets: [
      {id: 'mnist', 
        options: [
          {key: "0", value: '0'},
          {key: "1", value: '1'},
          {key: "2", value: '2'},
          {key: "3", value: '3'},
          {key: "4", value: '4'},
          {key: "5", value: '5'},
          {key: "6", value: '6'},
          {key: "7", value: '7'},
          {key: "8", value: '8'},
          {key: "9", value: '9'},
        ]
      },
      {id: 'gtsrb',
        options: [
          {key: "120km/hr", value: '120'},
          {key: "20km/hr", value: '20'},
          {key: "stop", value: 'stop'},
          {key: "Do Not Enter", value: 'Do Not Enter'},
        ]
      },
      {id: 'cifar',
        options: [
          {key: "Car", value: "car"},
          {key: "Bird", value: "bird"},
          {key: "Cat", value: "cat"},
          {key: "Deer", value: "deer"},
          {key: "Dog", value: "dog"},
          {key: "Frog", value: "frog"},
          {key: "Horse", value: "Horse"},
          {key: "Ship", value: "ship"},
          {key: "Truck", value: "truck"},
        ]
      },        
    ],
    key: "Select Target",
    value: "0",
  }),
  methods: {
	onClick(value){
		changeTarget(value)
	}
  },
  computed: {
    imageDropdown() {
      return (this.newDataset)
        ? this.datasets.find(x => x.id === this.newDataset)
        : this.datasets.find(x => x.id === "mnist")
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
