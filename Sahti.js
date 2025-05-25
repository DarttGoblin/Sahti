start_consulting.onclick = InitialiseChat;

yes.onclick = function () {
    CreateUserMessage('Yes');
    if (symptom_index == symptoms.length) {
        CreateBotMessage(`Unfortunatly you have been diagnosed by ${disease} disease! 😟💔`, false);
        ProvideMedication(disease);
        return;
    }
    CreateBotMessage(`Do you have ${symptoms[symptom_index].replace(/_/g, ' ')} symptom?`, true);
    symptom_index++;
}

no.onclick = function () {
    CreateUserMessage('No');
    DiagnoseSymptom().then(() => {
        symptom_index = 0;
        CreateBotMessage(`Do you have ${symptoms[symptom_index].replace(/_/g, ' ')} symptom?`, true);
        symptom_index = 1;
    });
}

async function DiagnoseSymptom() {
    return fetch('http://localhost:3001', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => response.json())
    .then(data => {
        if (Array.isArray(data) && data.length === 2) {
            disease = data[0].replace(/_/g, ' ');
            symptoms = data[1];
        }
        else {
            CreateBotMessage("Sorry for the inconvenience, there was an error retrieving information. Please refresh the page or try again later!", false);
            DisableButtons();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        CreateBotMessage("Sorry, there was a server error! The server might be undergoing maintenance. Please refresh or try again later.", false);
        DisableButtons();
    });
}

async function ProvideMedication(confirmed_disease) {
    try {
        const response = await fetch('http://localhost:3002', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ confirmed_disease }),
        });

        const data = await response.json();

        if (data.success) {
            await Wait(1500);
            await CreateBotMessage(data.treatment + ' 💊', false);
            await CreateBotMessage(data.recommendation + ' 💡', false);
            await CreateBotMessage("Please don't forget to download your prescription 📄", false);
            await CreatePrescriptionButton(confirmed_disease);
            DisableButtons();
        } else {
            console.error("Server error:", data.error);
            CreateBotMessage("Sorry for the inconvenience, there was an error retrieving information. Please refresh the page or try again later!", false);
            DisableButtons();
        }
    } catch (error) {
        console.error('Network error:', error);
        CreateBotMessage(`Sorry, I am unable to provide ${confirmed_disease} medication. There was a server error! The server might be undergoing maintenance. Please refresh or try again later.`, false);
        DisableButtons();
    }
}

function CreatePrescription(confirmed_disease) {
    fetch('http://localhost:3003', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ confirmed_disease }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.blob();
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'prescription.pdf';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    })
    .catch(error => {
        console.error('Error:', error);
        CreateBotMessage("Sorry, there was an error generating the prescription. Please try again later.", false);
        DisableButtons();
    });
}

const chatCon = document.getElementById("chatCon");
const searchEng = document.getElementById("searchEng");
const helpArrowIcon = document.getElementById("helpArrowIcon");

var placeHInterval = setInterval(GeneratePlaceHolders, 7000);
var suggetionsOff = true;

GeneratePlaceHolders();
CreateSuggetionsButtons();

searchEng.onkeydown = function(event) {if (event.key == 'Enter') {SendPrompt(searchEng.value);}}
helpArrowIcon.onclick = function() {SendPrompt(searchEng.value);}

function GenerateWaitSign() {
    const waitingSpan = document.createElement('span');
    const waitingIcon = document.createElement('i');
    const sender = document.createElement("span");

    sender.textContent = 'LogSpectrum';
    sender.classList.add('references');

    waitingIcon.classList.add('fa');
    waitingIcon.classList.add('fa-gear');
    waitingIcon.classList.add('waitingIconStyle');
    waitingIcon.classList.add('rotateAnimation');

    waitingSpan.appendChild(sender);
    waitingSpan.appendChild(waitingIcon);

    chatCon.appendChild(waitingSpan);
    return waitingSpan;
}
function CreateSuggetionsButtons() {
    const parentDiv = document.createElement('div');
    const logoImg = document.createElement('img');
    const buttonsCont = document.createElement('div');
    const childButtonOne = document.createElement('button');
    const childButtonTwo = document.createElement('button');
    const childButtonThree = document.createElement('button');
    const buttonIconOne = document.createElement('i'); 
    const buttonIconTwo = document.createElement('i'); 
    const buttonIcontThree = document.createElement('i'); 

    const ruleWritingTipsChoice = other_disease[Math.floor(Math.random() * other_disease.length)]
    const troubleshootingGuideChoice = another_disease[Math.floor(Math.random() * another_disease.length)]
    const securityBestPracticesChoice = last_disease[Math.floor(Math.random() * last_disease.length)]

    logoImg.src = 'HelpMedia/logoImg.png';

    parentDiv.classList.add('parentDivStyle');
    logoImg.classList.add('logoImgStyle');
    buttonsCont.classList.add('buttonsContStyle');
    childButtonOne.classList.add('childButtonStyle');
    childButtonTwo.classList.add('childButtonStyle');
    childButtonThree.classList.add('childButtonStyle');

    buttonIconOne.classList.add('fa');
    buttonIconTwo.classList.add('fa');
    buttonIcontThree.classList.add('fa');
    buttonIconOne.classList.add('fa-heart-pulse');  
    buttonIconTwo.classList.add('fa-stethoscope');
    buttonIcontThree.classList.add('fa-notes-medical'); 

    buttonIconOne.id = 'ruleWritingIcon';
    buttonIconTwo.id = 'troubleshootingIcon';
    buttonIcontThree.id = 'securityIcon';

    childButtonOne.appendChild(buttonIconOne);
    childButtonTwo.appendChild(buttonIconTwo);
    childButtonThree.appendChild(buttonIcontThree);

    childButtonOne.innerHTML += '<br><br>' + ruleWritingTipsChoice; 
    childButtonTwo.innerHTML += '<br><br>' + troubleshootingGuideChoice; 
    childButtonThree.innerHTML += '<br><br>' + securityBestPracticesChoice;

    buttonsCont.appendChild(childButtonOne);
    buttonsCont.appendChild(childButtonTwo);
    buttonsCont.appendChild(childButtonThree);

    parentDiv.appendChild(logoImg);
    parentDiv.appendChild(buttonsCont);

    chatCon.appendChild(parentDiv);    

    childButtonOne.onclick = function() {SendPrompt(ruleWritingTipsChoice);}
    childButtonTwo.onclick = function() {SendPrompt(troubleshootingGuideChoice);}
    childButtonThree.onclick = function() {SendPrompt(securityBestPracticesChoice);}
}
function GenerateUserChatSpan(chatSpanContent) {
    var chatSpan = document.createElement("span");
    var sender = document.createElement("span");
    chatSpan.textContent = chatSpanContent;
    sender.textContent = 'User';
    chatSpan.classList.add('chatSpans');
    chatSpan.classList.add('prompts');
    sender.classList.add('references');
    chatCon.appendChild(sender);
    chatCon.appendChild(chatSpan);
    chatCon.scrollTop = chatCon.scrollHeight;
    searchEng.value = '';
}
function GenerateAIChatSpan(chatSpanContent) {
    var chatSpan = document.createElement("span");
    var sender = document.createElement("span");
    sender.textContent = 'LogSpectrum';
    chatSpan.classList.add('chatSpans');
    chatSpan.classList.add('aiResp');
    sender.classList.add('references');
    chatCon.appendChild(sender);
    chatCon.appendChild(chatSpan);
    chatCon.scrollTop = chatCon.scrollHeight;
    searchEng.value = '';
    var chatSpanContentArray = chatSpanContent.split("");
    for (var i = 0; i < chatSpanContentArray.length; i++) {
        (function(index) {
            setTimeout(() => { chatSpan.textContent += chatSpanContentArray[index]; }, 10 * index);
        })(i);
    }
}
function SendPrompt(prompt) {
    if (prompt == '') {return;}
    searchEng.disabled = true;
    if (suggetionsOff) {
        chatCon.innerHTML = '';
        chatCon.style.display = 'block';
        suggetionsOff = false;
    }
    GenerateUserChatSpan(prompt);
    const waitingSpan = GenerateWaitSign();
    fetch('http://localhost:8014', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            chatCon.removeChild(waitingSpan);
            GenerateAIChatSpan(data.aiResponse);
            searchEng.disabled = false;
        }
        else {
            chatCon.removeChild(waitingSpan);
            console.log(data.error);
            alert("An error has been occured! Try again later.");
            searchEng.disabled = false;
        }
    })
    .catch(error => {
        chatCon.removeChild(waitingSpan);
        console.log(error);
        alert("An error has been occured! Try again later.");
        searchEng.disabled = false;
    });
}
function GeneratePlaceHolders() {
    var networkConcept = heart_questions[Math.floor(Math.random() * (heart_questions.length))];
    var networkConceptArray = networkConcept.split("");
    searchEng.placeholder = "";
    for (var i = 0; i < networkConceptArray.length; i++) {
        (function(index) {
            setTimeout(() => { searchEng.placeholder += networkConceptArray[index]; }, 50 * index);
        })(i);
    }
}

const bot_assistant = document.querySelector('.bot-assistant');

bot_assistant.onlick = function() {
    chatCon.scrollTop = chatCon.scrollHeight;
}