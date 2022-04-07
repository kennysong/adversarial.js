<template>
  <div>
    <canvas id="original" class="tableimg" max-width="224" height="224"></canvas>
    <img id = "frame" src="" class = "img-fluid" alt = "responsive">
  </div>
</template>

<script>
//notes: use getImg(), loadImage(), and drawImg() in intro.js as a base for scripts
export default {
  name: 'ImageContainer',
  methods: {
  }
}
export function updateImage() {
  var frame = document.getElementById("frame")
  var source = document.getElementById("fileid")
  frame.src = URL.createObjectURL(source.files[0])
}
</script>

<style>

</style>