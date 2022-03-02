<template>
  <div class="banner">
    <div class="headers">Adversarial Learning</div>
    <div class="headers page-title">-</div>
    <div class="headers page-title">{{ pageTitle }}</div>
    <div class="links">
      <a href="#">Home</a>
      <a href="#">FAQ</a>
      <a href="#">Demo</a>
      <a href="#">API</a>
      <a href="#">GitHub</a>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Navigation',
  props: {
    pageTitle: String
  }
}
</script>

<style>
.banner {
  display: flex;
  width: 100%;
  height: 100px;
  background-color: #D3DAE4;
  padding-left: 35px;
}

.headers {
  display: inline-block;
  color: #333333;
  font-weight: bolder;
  font-family: "Raleway-italic";
  font-size: 28pt;
  margin-top: 25px;
}

.page-title {
  margin-left: 10px;
}

.links {
  display: inline-block;
  margin-top: 40px;
  margin-left: auto;
  margin-right: 35px;
}

a {
  color: #333333;
  font-family: "Montserrat";
  text-decoration: none;
  margin-right: 35px;
}

a:hover {
  color: #0C68BE;
}

a:active {
  font-weight: bold;
}
</style>

