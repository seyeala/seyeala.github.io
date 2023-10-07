document.querySelector(".menu-title").addEventListener("click", function() {
    var dropdown = document.querySelector(".menu-dropdown");
    dropdown.style.display = dropdown.style.display === "block" ? "none" : "block";
});
