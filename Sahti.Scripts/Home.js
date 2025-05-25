const get_care = document.querySelector('.get-care');
const read_report = document.querySelector('.read-report');

get_care.onclick = function() {
    section[1].scrollIntoView({behavior: 'smooth'});
}

read_report.onclick = function() {
    window.open("https://drive.google.com/file/d/18RnDnL3j3Qy1phvupXgbJ8FQUFJROrbA/view?usp=drive_link", "_blank")
}