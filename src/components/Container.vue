<template>
  <div class="wrap">
    <div class="options">
      <div class="label">Select a Model</div>
      <div class="drop-down"><ModelDropDown/></div>
      <div class="label">Select a Dataset</div>
      <div class="drop-down"><DatasetDropDown @changeDataset = "updateDropdowns"/></div>
      <div class="label">Select an Attack</div>
      <div class="drop-down"><AttackDropDown/></div>
      <div class="label">Turn This Image Into a/an:</div>
      <div class="drop-down"><ImageDropDown :newDataset="newDataset"/></div>
    </div>
    <div class="imageContainer">
      <ImageContainer />
      <PredictionData />
    </div>
    <div class="buttonDiv">
      <div class="generate">
        <Button value = "Run Neural Network" description="predict"/>
        <br>
        <Button value = "Generate" description="adv"/>
      </div>
      <div class="upload">
        <Button value = "Upload Image" description="upload-image"/>
        <br>
        <Button value = "Next Image" description="next-image"/>
        <br>
        <Button value = "Download Image"/>
      </div>
    </div>
  </div>
</template>

<script>
import Button from './Button.vue'
import DatasetDropDown from './DatasetDropDown.vue'
import ModelDropDown from './ModelDropDown.vue'
import AttackDropDown from './AttackDropDown.vue'
import ImageDropDown from './ImageDropDown.vue'
import ImageContainer from './ImageContainer.vue'
import PredictionData from './PredictionData.vue'

export default {
  name: 'Container',
  components: {
    Button,
    ModelDropDown,
    DatasetDropDown,
    AttackDropDown,
    ImageDropDown,
    ImageContainer,
    PredictionData
  },
  methods: {
    updateDropdowns(val) {
      this.newDataset = val
    }
  },
  data() { return { newDataset: "mnist" } }
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
  max-width: 1300px;
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



.buttonDiv {
  width:100%;
  display:flex;
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


  .button {
  width:80%;
  height:4em;
  font-size:12px;
}


  .buttondiv{
    width:50%;
    display:flex;

  }
  .generate {
    width: 50%;
    height: fit-content;
    padding: 1em;
    margin:0em;
  }

  .imageContainer {
    height: fit-content;
    width: 100%;
    height: 100%;
    object-fit:contain;
    background-color: rgb(255, 255, 255);
    padding:1em
  }

  .upload {
    width: 50%;
    height: fit-content;
    padding: 1em;
  }

  .label {
    text-align: left;
    padding-top: .8em;
    padding-bottom: .3em;
  }


  .dropdown-button {
    height:3em;
  }


  .drop-down {
    width: 100%;
    text-align: center;
  }
  .dropdown-button {
    width: 70%;
    text-align: center;
  }

  
}
</style>

