<template>
  <div class="dropdown">
<<<<<<< HEAD
  <select id="select-model">
            <option value="" selected disabled>Select model</option>
            <option value="mnist">MNIST (digit recognition)</option>
            <option value="gtsrb">GTSRB (street sign recognition)</option>
            <option value="cifar">CIFAR-10 (object recognition, small)</option>
            <option value="imagenet">ImageNet (object recognition, large)</option>
          </select>
  </div>
</template>
=======
  <button class="btn btn-secondary dropdown-toggle" type="button" id="dropdownMenuButton1" data-bs-toggle="dropdown" aria-expanded="false">
    Select Model
  </button>
  <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton1">
        <li><a class="dropdown-item" href="#">MNIST (digit recognition)</a></li>
        <li><a class="dropdown-item" href="#">GTSRB (street sign recognition)</a></li>
        <li><a class="dropdown-item" href="#">CIFAR-10 (object recognition, small)</a></li>
        <li><a class="dropdown-item" href="#">ImageNet (object recognition, large)</a></li>
        </ul>
</div>
  </template>
>>>>>>> 20105927ea96456ae4515245201a2c1817266166

<script>
export default {
  name: 'ModelDropDown',
  props: {
    description: String
  }
}
</script>

<style>

</style>