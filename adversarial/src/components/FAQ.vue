<template>
  <div>
    <Accordion />
  </div>
</template>

<script>
import Accordion from './Accordion.vue'
export default {
  name: 'FAQ',
  props: {
    description: String
  },
  components: {
    Accordion
  }
}
</script>

<style>

</style>