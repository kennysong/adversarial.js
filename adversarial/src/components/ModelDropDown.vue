<template>
  <div class="dropdown">
      <button class="btn btn-primary dropdown-toggle" type="button" id="select-model" data-bs-toggle="dropdown" aria-expanded="false">
      {{id}}
      </button>
        <ul class="dropdown-menu" aria-labelledby="select-model">
        <li v-for="option in options" :key="option">
          <a class="dropdown-item" @click="id = option.id; value = option.value" href="javascript: void(0)">{{option.id}}</a>
        </li>
        </ul>
      </div>
  </template>
<script>
import "bootstrap/dist/css/bootstrap.min.css"
import "bootstrap"
export default {
  name: 'ModelDropDown',
  props: {
    description: String
  },
  data: () => ({
    options: [
      { id: 'MNIST (digit recognition)', value: 'mnist' },
      { id: 'GTSRB (street sign recognition)', value: 'gtsrb' }, 
      { id: 'CIFAR-10 (object recognition, small)', value: 'cifar10' }, 
      { id: 'ImageNet (object recognition, large)', value: 'imagenet'}
      ],
    id: 'Select Model',
    value: ''
  }),
}
</script>

<style>

</style>