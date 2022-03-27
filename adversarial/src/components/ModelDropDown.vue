<template>
  <div class="dropdown">
      <button class="btn btn-primary dropdown-toggle" type="button" id="ModelDropDown" data-bs-toggle="dropdown" aria-expanded="false">
      {{value}}
      </button>
        <ul class="dropdown-menu" aria-labelledby="ModelDropDown">
        <li v-for="option in options" :key="option">
          <a class="dropdown-item" @click="value = option" href="javascript: void(0)">{{option}}</a>
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
    options: ['MNIST (digit recognition)', 'GTSRB (street sign recognition)', 'CIFAR-10 (object recognition, small)', 'ImageNet (object recognition, large)'],
    value: 'Select Model'
  }),
}
</script>

<style>

</style>