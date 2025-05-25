const nav_links = document.querySelectorAll('.nav-links');
const section = document.getElementsByTagName('section');
const sign_in = document.querySelector('.sign-in');
const settings = document.querySelector('.settings');
const responsive_nav = document.querySelector('.responsive-nav');
const closing_icon = document.querySelector('.closing-icon');
const responsive_nav_link = document.querySelectorAll('.responsive-nav-link');
const responsive_sign_in = document.querySelector('.responsive-sign-in');

nav_links.forEach((link, index) => {
    link.onclick = function() {
        section[index].scrollIntoView({behavior: 'smooth'});
    } 
});

sign_in.onclick = function() {
    alert('This service is not available now! Check later.');
};

responsive_nav_link.forEach((link, index) => {
    link.onclick = function() {
        section[index].scrollIntoView({behavior: 'smooth'});
        responsive_nav.style.display = 'none';
    } 
});

sign_in.onclick = responsive_sign_in.onclick = function() {
    alert('This service is not available now! Check later.');
};

settings.onclick = function() {
    responsive_nav.style.display = 'flex';
}

closing_icon.onclick = function() {
    responsive_nav.style.display = 'none';
}