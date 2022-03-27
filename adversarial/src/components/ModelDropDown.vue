<template>
  <div class="dropdown">
      <button class="btn btn-primary dropdown-toggle" type="button" id="select-model" data-bs-toggle="dropdown" aria-expanded="false" value = "">
      {{key}}
      </button>
        <ul class="dropdown-menu" aria-labelledby="select-model">
        <li v-for="option in options" :key="option.key">
          <a class="dropdown-item" @click="key = option.key; value = option.value" href="javascript: void(0)">{{option.key}}</a>
        </li>
        </ul>
      </div>
  </template>
<script>
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
      { key: "MNIST (digit recognition)", value: 'mnist' },
      { key: "GTSRB (street sign recognition)", value: 'gtsrb' }, 
      { key: "CIFAR-10 (object recognition, small)", value: 'cifar10' }, 
      { key: "ImageNet (object recognition, large)", value: 'imagenet'}
      ],
    key: "Select Model",
    value: ""
  }),
}
</script>

<style>

</style>