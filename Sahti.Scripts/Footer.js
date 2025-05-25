const footer_icon = document.querySelectorAll('.footer-icon');
const footer_body_info = document.querySelectorAll('.footer-body-info');
const quick_link = document.querySelectorAll('.quick-link');
const professor = document.querySelector('.professor');
const footer_body_support_btn = document.querySelector('.footer-body-support-btn');
const lets_talk_contact_btn = document.querySelector('.lets-talk-contact-btn');

const social_media = [
    "not_available",
    "not_available",
    "https://www.linkedin.com/in/yassine-bazgour-178b73305/",
    "https://www.linkedin.com/in/zahra-abouhane-27b8b218a/",
]

professor.onclick = function() {
    window.open('https://www.linkedin.com/in/hasna-elalaoui-7a403942/', '_blank');
}

footer_body_info.forEach((info, index) => {
    info.onclick = function() {
        const text = this.textContent;
        if (index == 1 || index == 2) {
            window.open(text, '_blank');
            return;
        }
        else {
            navigator.clipboard.writeText(text).then(() => {
                alert(`${info.id} has been copied to clipboard:`);
            }).catch(err => {
                alert('Failed to copy text:', err);
            });
        }
    };
});

footer_icon.forEach((icon, index) => {
    icon.onclick = function() {
        if (social_media[index] == 'not_available') {
            alert('This service is not available now! Check later.');
            return;
        }
        window.open(social_media[index], '_blank');}
});

quick_link.forEach(link => {
    link.onclick = function() {
        alert('This service is not available now! Check later.');
    }
});

footer_body_support_btn.onclick = lets_talk_contact_btn.onclick = function() {
    alert('This service is not available now! Check later.');
}