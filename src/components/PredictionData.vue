<template>
  <table class="predictionTable">
    <tr>
      <th>Original</th>
      <th>New</th>
    </tr>
    <tr>
      <td class="data"> Prediction: <div id="prediction"></div></td>
      <td class="data"> Prediction: <div id="prediction-adv"></div></td>
    </tr>
    <tr>
      <td class="data"> Probability: <div id="prediction-status"></div></td>
      <td class="data"> Probability: <div id="prediction-adv-status"></div></td>
    </tr>
  </table>   
</template>

<script>
export default {
  name: 'PredictionData'
}
</script>

<style>
table.predictionTable {
  position: relative;
  border: none;
  width: 90%;
  height: 7em;
  margin: 5em 1em 0em 1em;
  font-size: 14px;
  font-weight: bold;
  text-align: center;
}

table.predictionTable td.button {
  width: 60%;
}

table.predictionTable th {  
  width: 20%;
  font-family: "Raleway-italic";
}

table.predictionTable td.data {
  text-align: left;
  font-family: "Raleway";
  vertical-align: top;
}

 @media (max-width: 600px) {

.prediction {
  position: fixed;
  bottom: 70px;
  width: 100%;
  height: 6em;
  margin:1em;
}
  

table.predictionTable {
  table-layout: fixed;
  border: none;
  width: 100%;
  height: 10em;
  font-size: 12px;
  font-weight: bold;
  text-align: center;
}
  
}

</style>