<template>
  <div class="dropdown">
    <button class="dropdown-button dropdown-toggle" type="button" id="dropdownMenuButton1" data-bs-toggle="dropdown" aria-expanded="false">
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

.dropdown-toggle::after {
  float: right !important;
  vertical-align: middle !important;
}
</style>