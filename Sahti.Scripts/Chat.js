const messages_container = document.querySelector('.messages-container');
const bot_buttons_container = document.querySelector('.bot-buttons-container');
const start_consulting = document.querySelector('.start-consulting');
const chat_bot_image = document.querySelector('.chat-bot-image');
const bot_buttons = document.querySelectorAll('.bot-buttons');
const yes = document.querySelector('.yes');
const no = document.querySelector('.no');

var disease;
var symptoms;
var symptom_index;

function InitialiseChat() {
    const welcome_user = "I'm here to help you understand and manage your digestive health better. Let's begin by assessing your symptoms. Please answer the following questions honestly to get personalized insights.";

    DisableButtons();
    messages_container.style.display = 'flex';
    bot_buttons_container.style.display = 'flex';
    start_consulting.style.display = 'none';
    chat_bot_image.style.display = 'none';

    const bot_message = document.createElement('span');
    const message_to_letter = welcome_user.split("");
    bot_message.classList.add('message');
    bot_message.classList.add('bot-message');
    messages_container.appendChild(bot_message);
    messages_container.scrollTo({ top: messages_container.scrollHeight, behavior: 'smooth' });
    
    for (var i = 0; i < message_to_letter.length; i++) {
        (function (index) {
            setTimeout(() => { bot_message.textContent += message_to_letter[index]; }, 20 * index);
        })(i);
    }

    DiagnoseSymptom().then(() => {
        setTimeout(() => {
            CreateBotMessage(`Do you have ${symptoms[0].replace(/_/g, ' ')} symptom?`, true);
            symptom_index = 1;
        }, 4000);
    });
}

async function CreateBotMessage(message, enable) {
    const bot_message = document.createElement('span');
    const message_to_letter = message.split("");
    bot_message.classList.add('message');
    bot_message.classList.add('bot-message');
    messages_container.appendChild(bot_message);
    
    
    for (var i = 0; i < message_to_letter.length; i++) {
        (function (index) {
            setTimeout(() => {
                bot_message.textContent += message_to_letter[index]; 
                messages_container.scrollTo({ 
                    top: messages_container.scrollHeight,
                    behavior: 'smooth' 
                });
            }, 20 * index);
        })(i);
    }
    
    const message_length = message_to_letter.length;
    await Wait(20 * message_length);

    if (enable) {EnableButtons();}
}

function CreateUserMessage(answer) {
    const user_message = document.createElement('span');
    user_message.classList.add('message');
    user_message.classList.add('user-message');
    user_message.textContent = answer;
    messages_container.appendChild(user_message);
    messages_container.scrollTo({ top: messages_container.scrollHeight, behavior: 'smooth' });
    DisableButtons();
}

function DisableButtons() {
    bot_buttons.forEach(button => {
        button.style.backgroundColor = 'rgb(100, 100, 100)';
        button.style.color = 'rgb(255, 255, 255)';
        button.disabled = true;
    });
}

async function EnableButtons() {
    bot_buttons[0].style.backgroundColor = 'rgb(255, 255, 255)';
    bot_buttons[0].style.color = 'rgb(0, 0, 0)';
    bot_buttons[1].style.backgroundColor = 'rgb(98, 64, 232)';
    bot_buttons[1].style.color = 'rgb(255, 255, 255)';
    bot_buttons[0].disabled = false;
    bot_buttons[1].disabled = false;
}

function Wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function CreatePrescriptionButton(confirmed_disease) {
    const pre_button = document.createElement('button');
    pre_button.classList.add('pre-button');
    pre_button.innerHTML = 'Download Prescription 💾';
    messages_container.appendChild(pre_button);
    messages_container.scrollTo({ top: messages_container.scrollHeight, behavior: 'smooth' });

    pre_button.onclick = function() {
        CreatePrescription(confirmed_disease);
    }
}