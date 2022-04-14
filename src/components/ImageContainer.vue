<template>
  <div>
    <canvas id="original" class="tableimg" max-width="224" height="224"></canvas>
    <img id = "frame" src="" class = "img-fluid" alt = "responsive" data-bs-toggle="modal" data-bs-target="#compareModal">
    <div class="modal fade" id="compareModal" tabindex="-1" aria-labelledby="compareModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-dialog-centered modal-dialog-scrollable modal-xl">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="compareModalLabel">Compare the Images!</h5>
           <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <div class = "row g-1">
              <div class = "col">
              <img id = "frame2" src="" class = "img-fluid">
              </div>
              <div class = "col">
              <img id = "frame3" src="" class = "img-fluid">
              </div>
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
            <button type="button" class="btn btn-primary">Save changes</button>
          </div>
        </div>
      </div>
    </div>
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
  var frame2 = document.getElementById("frame2")
  var frame3 = document.getElementById("frame3")
  var source = document.getElementById("fileid")
  frame.src = URL.createObjectURL(source.files[0])
  frame2.src = frame.src
  frame3.src = frame.src
}
</script>

<style>
</style>
