<template>
  <div class="footer">
    <table class="footerTable">
      <tr>
        <td rowspan="5" class="button"><Button value = "Prediction"/></td>
      </tr>
      <tr>
        <th>Original</th>
        <th>New</th>
      </tr>
      <tr>
        <td class="info"> Prediction: </td>
        <td class="info"> Prediction: </td>
      </tr>
      <tr>
        <td class="info"> Probability: </td>
        <td class="info"> Probability: </td>
      </tr>
    </table>
  </div>
</template>

<script>
import Button from './Button.vue'
export default {
  name: 'Footer',
  components: {
    Button
  }
}
</script>

<style>
.footer {
  background-color: #F4F6F9;
  position: absolute;
  bottom: 0px;
  width: 100%;
  height: 7em;
}

table.footerTable {
  border: none;
  width: 100%;
  height: 100%;
  font-size: 20px;
  font-weight: bold;
  text-align: center;
}

table.footerTable td.button {
  width: 50%;
}

table.footerTable th {  
  width: 25%;
  font-family: "Raleway-italic";
}

table.footerTable td.info {
  text-align: left;
  font-family: "Raleway";
}
</style>