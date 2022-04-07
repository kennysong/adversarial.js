<template>
	<div class="wrapper">
			<input class="button" type="button" v-model=value @click = "onClick()">
		<div class="uploader">
			<input id="fileid" type="file" accept="image/*" @change = "upload()" hidden/>
		</div>
	</div>
</template>

<script>
import {nextImage, uploadImage, predictImg, attack} from "../../public/js/intro.js"
export default {
  name: 'Button',
  props: {
    description: String,
    value: String
  },
  methods: {
    onClick(){
		if(this.description == "next-image"){nextImage()}
        else if(this.description == "predict") {predictImg()}
        else if(this.description == "adv"){attack()}
		else if(this.description == "upload-image"){document.getElementById('fileid').click()}
	},
	upload(){
		uploadImage()
	}
  }
}
</script>

<style>
.button {
  background-color: #EBF2FC;
  color: #000000;
  font-family: "Raleway-italic";
  font-weight: bold;
  font-size: 20px;
  width: 12em;
  height: 2em;
  box-shadow: 1px 2px 4px #8c8c8c;
  border-radius: 0;
  border: 0;
}
</style>