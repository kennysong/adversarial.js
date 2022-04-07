<template>
  <div class="footer">
    (c) Senior Design Adversarial Team
  </div>      
</template>

<script>
export default {
  name: 'Footer'
}
</script>

<style>
.footer {
  background-color: #F4F6F9;
  position: relative;
  /* position: absolute;
  bottom: 0px; */
  width: 100%;
  height: 2em;
  padding: .5em;
  margin-top: 4em;
  font-size: 20px;
  font-weight: bold;
  font-family: "Raleway";
  text-align: center;
}

@media (max-width: 600px) {
  .footer {
    position: fixed;
    bottom: 70px;
    width: 100%;
    height: 2em;
    font-size: 12px;
    font-weight: bold;
    text-align: center;
  }
}
</style>