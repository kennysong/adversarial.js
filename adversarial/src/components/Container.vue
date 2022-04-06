<template>
  <div class="wrap">
    <div class="options">
      <div class="label">Select a Model</div>
      <div class="drop-down"><ModelDropDown/></div>
      <div class="label">Select a Dataset</div>
      <div class="drop-down"><DatasetDropDown/></div>
      <div class="label">Select an Attack</div>
      <div class="drop-down"><AttackDropDown/></div>
      <div class="label">Turn This Image Into a/an:</div>
      <div class="drop-down"><ImageDropDown/></div>
    </div>
    <div class="imageContainer">
      <ImageContainer />
    </div>
    <div class="generate">
      <Button value = "Generate"/>
    </div>
    <div class="upload">
      <Button value = "Upload Image" description="upload-image"/>
      <br>
      <Button value = "Next Image" description="next-image"/>
    </div>
  </div>
</template>

<script>
import Button from './Button.vue'
import ModelDropDown from './ModelDropDown.vue'
import AttackDropDown from './AttackDropDown.vue'
import ImageDropDown from './ImageDropDown.vue'
import ImageContainer from './ImageContainer.vue'
import DatasetDropDown from './DatasetDropDown.vue'
export default {
  name: 'Container',
  components: {
    Button,
    ModelDropDown,
    DatasetDropDown,
    AttackDropDown,
    ImageDropDown,
    ImageContainer
  }
}
</script>

<style>
.mainImage {
  max-width: 100%;
  height: auto;
}

.wrap {
  background-color: #FFFFFF;
  width: 85%;
  height: fit-content;
  margin: auto;
  margin-top: 4em;
  padding: 1em;
  display: flex;
  flex-flow: row wrap;
  text-align: center;
}

.options {
  width: 45%;
  height: fit-content;
  padding: 1em;
  text-align: left;
}

.generate {
  width: 45%;
  height: fit-content;
  padding: 1em;
}

.imageContainer {
  width: 55%;
  height: fit-content;
  padding-top: 4em;
}

.upload {
  width: 55%;
  height: fit-content;
  padding: 1em;
}

.label {
  text-align: left;
  font-weight: 600;
  font-family: "Raleway";
  padding-top: 1em;
  padding-bottom: .5em;
}

.drop-down {
  margin: auto;
  text-align: center;
}

@media (max-width: 600px) {

  .mainImage {
    max-width: 100%;
    height: auto;
  }

  .label {
    display: inline-block;
    vertical-align: top;
    padding-top: 25px;
  }

  .wrap {
    width: 85%;
    height: fit-content;
    margin: auto;
    margin-top: .5em;
    padding: .1em;
    display: flex;
    flex-direction: column;
    flex-flow: row wrap;
    text-align: center;
  }

  .options {
    width: 100%;
    height: fit-content;
    padding: .5em;
    text-align: center;
    
    
  }

  .generate {
    width: 45%;
    height: fit-content;
    padding: 1em;
  }

  .imageContainer {
    height: fit-content;
    width: 100%;
    height: 100%;
    object-fit:contain;
    background-color: rgb(255, 255, 255);
    margin-top: auto;
  }

  .upload {
    width: 55%;
    height: fit-content;
    padding: 1em;
  }

  .label {
    text-align: left;
    padding-top: .8em;
    padding-bottom: .3em;
  }
}
</style>

