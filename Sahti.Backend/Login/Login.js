const mysql = require("mysql");
const express = require("express");
const cors = require("cors");

const app = express();
const port = 8008;

app.use(express.json());
app.use(cors({ origin: "*" }));

app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
    next();
});

app.post("/", (req, res) => {
    const givenusername = req.body.givenusername;
    const givenpassword = req.body.givenpassword;
    let resObject = { success: false };
    var supervisorData;

    const connection = mysql.createConnection({
        host: 'localhost',
        user: 'root',
        password: '',
        database: 'pfedb'
    });

    connection.connect((err) => {
        if (err) {
            console.error('Error connecting to MySQL database: ' + err.stack);
            resObject.error = "Error connecting to the database";
            res.status(500).json(resObject);
            return;
        }
        console.log('Connected to MySQL database with connection id ' + connection.threadId);

        connection.query('SELECT * FROM Supervisor WHERE username = ? AND passw = ?', [givenusername, givenpassword], (error, results) => {
            if (error) {
                console.log('Error ' + error.stack);
                res.status(500).json({ success: false, error: error.stack });
                return;
            }
            if (results.length > 0) {
                supervisorData = results[0];
                res.status(200).json({ success: true, supervisorData });
            }
            else {res.status(200).json({ success: false, error: "No User Has Been Found" });}
        });
    });
});

app.listen(port, () => console.log("Listening on port " + port));
