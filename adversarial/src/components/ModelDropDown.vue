<template>
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