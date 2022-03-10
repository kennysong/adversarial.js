<template>
  <div class="dropdown">
  <button class="btn btn-primary dropdown-toggle" type="button" id="dropdownMenuButton1" data-bs-toggle="dropdown" aria-expanded="false">
    Select Attack
  </button>
  <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton1">
        <li><a class="dropdown-item" href="#">Carlini & Wagner (strongest) (digit recognition)</a></li>
        <li><a class="dropdown-item" href="#">Jacobian-based Saliency Map Attack</a></li>
        <li><a class="dropdown-item" href="#">Basic Iterative Method (stronger)</a></li>
        <li><a class="dropdown-item" href="#">Fast Gradient Sign Method (weak)</a></li>
        </ul>
</div>
  </template>

<script>
export default {
  name: 'AttackDropDown',
  props: {
    description: String
  }
}
</script>

<style>

</style>