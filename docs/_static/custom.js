// custom.js

document.addEventListener('DOMContentLoaded', function () {
    // Get all links that start with http:// or https://
    var externalLinks = document.querySelectorAll('a[href^="http://"], a[href^="https://"]');

    // Loop through each link and add the target attribute
    externalLinks.forEach(function (link) {
        link.setAttribute('target', '_blank');
    });
});