<template>
  <input class="btn btn-primary" type="button">
</template>

<script>
export default {
  name: 'Button',
  props: {
    description: String
  }
}
</script>

<style>
.button {
  background-color: #EBF2FC;
  color: #000000;
  font-family: "Raleway-italic";
  font-weight: bold;
  width: 15em;
  border-radius: 0;
  border: 0;
}
</style>