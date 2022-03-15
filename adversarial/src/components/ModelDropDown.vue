<template>
  <div class="dropdown">
  <select id="select-model">
            <option value="" selected disabled>Select model</option>
            <option value="mnist">MNIST (digit recognition)</option>
            <option value="gtsrb">GTSRB (street sign recognition)</option>
            <option value="cifar">CIFAR-10 (object recognition, small)</option>
            <option value="imagenet">ImageNet (object recognition, large)</option>
          </select>
  </div>
</template>

<script>
export default {
  name: 'ModelDropDown',
  props: {
    description: String
  }
}
</script>

<style>

</style>