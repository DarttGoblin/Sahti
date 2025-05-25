const mysql = require("mysql");
const express = require("express");
const cors = require("cors");

const app = express();
const port = 8010;

app.use(express.json());
app.use(cors({ origin: "*" }));

app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
    next();
});

app.post("/", (req, res) => {
    let resObject = { success: false };
    var logData;

    const connection = mysql.createConnection({
        host: 'localhost',
        user: 'root',
        password: '',
        database: 'PFEdb'
    });

    connection.connect((err) => {
        if (err) {
            console.error('Error connecting to MySQL database: ' + err.stack);
            resObject.error = "Error connecting to the database";
            res.status(500).json(resObject);
            return;
        }
        console.log('Connected to MySQL database with connection id ' + connection.threadId);

        connection.query('SELECT * FROM Log', (logError, logResults) => {
            if (logError) {
                console.log('Error: ' + logError.stack);
                res.status(500).json({ success: false, error: logError.stack });
                return;
            }
            logData = logResults;
            connection.query('SELECT * FROM Supervisor', (superVError, superVResults) => {
                if (superVError) {
                    console.error('Error: ' + superVError.stack);
                    res.status(500).json({ success: false, error: superVError.stack });
                    return;
                }
                supervisorData = superVResults;
                res.status(200).json({ success: true, logData, supervisorData});
            });
        });
    });
});


app.listen(port, () => console.log("Listening on port " + port));