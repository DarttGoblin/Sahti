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