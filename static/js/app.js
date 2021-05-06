$(function () {
    $('[data-toggle="tooltip"]').tooltip()
})


var slider = document.getElementById("complianceRange");
var output = document.getElementById("range_val");
output.innerHTML = slider.value; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
slider.oninput = function() {
    output.innerHTML = this.value;
}