<template>
  <div class="footer">
    <table class="footerTable">
      <tr>
        <td rowspan="4" class="button"><Button value = "Prediction"/></td>
            <Button value = "Run Neural Network" id = "predict-original"/>
            <button id="predict-adv" class="button-primary" disabled>Click "Generate" First</button>
      </tr>
      <tr>
        <th>Original</th>
        <th>New</th>
      </tr>
      <tr>
        <td class="info"> Prediction: <div id="prediction"></div></td>
        <td class="info"> Prediction: <div id="prediction-adv"></div></td>
      </tr>
      <tr>
        <td class="info"> Probability: <div id="prediction-status"></div></td>
        <td class="info"> Probability: <div id="prediction-adv-status"></div></td>
      </tr>
    </table>
  </div>
            
            
</template>

<script>
import Button from './Button.vue'

export default {
  name: 'Footer',
  props: {
    description: String
  },
  components: {
    Button
  }
}
</script>

<style>
.footer {
  background-color: #F4F6F9;
  /* position: relative; */
  position: absolute;
  bottom: 0px;
  width: 100%;
  height: 7em;
  margin-top: 5em;
}

table.footerTable {
  border: none;
  width: 90%;
  height: 75%;
  font-size: 20px;
  font-weight: bold;
  text-align: center;
}

table.footerTable td.button {
  width: 60%;
}

table.footerTable th {  
  width: 20%;
  font-family: "Raleway-italic";
}

table.footerTable td.info {
  text-align: left;
  font-family: "Raleway";
  vertical-align: top;
}
</style>