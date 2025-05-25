const backgrounds = document.querySelectorAll('.background');
const about_links = document.querySelectorAll('.about-links');

for (i = 0; i < backgrounds.length; i++) {
    backgrounds[i].onclick = about_links[i].onclick = function() {
        alert('Something went wrong');
        return;
    }
}