<template>
  <div>
    <canvas id="original" class="tableimg" max-width="224" height="224"></canvas>
    <img id = "frame" src="" class = "img-fluid" alt = "responsive">
  </div>
</template>

<script>
function updateImage() {
  var frame = document.getElementById("frame")
  frame.src = URL.createObjectURL(frame.files[0])
}
//notes: use getImg(), loadImage(), and drawImg() in intro.js as a base for scripts
export default {
  name: 'ImageContainer',
  methods: {
    updateImage
  }
}
</script>

<style>

</style>