const mysql = require("mysql");
const express = require("express");
const cors = require("cors");

const app = express();
const port = 8012;

app.use(express.json());
app.use(cors({ origin: "*" }));

app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
    next();
});

app.post("/", (req, res) => {
    const newUserData = req.body.newUserData; 
    
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

        connection.query('SELECT * FROM Supervisor WHERE username = ?', [newUserData.username], (error, results) => {
            if (error) {
                console.log('Error ' + error.stack);
                res.status(500).json({ success: false, error: error.stack });
                return;
            }
            if (results.length > 0) {
                const repley = false;
                res.status(500).json({ success: true, repley });
                return;
            }
            const insertQuery = 'INSERT INTO Supervisor(fname, lname, email, username, passw) VALUES(?, ?, ?, ?, ?)';
            connection.query(insertQuery, [newUserData.fname, newUserData.lname, newUserData.email, newUserData.username, newUserData.password], (error) => {
                if (error) {
                    console.log('Error ' + error.stack);
                    res.status(500).json({ success: false, error: error.stack });
                    return;
                }
                const repley = true;
                res.status(500).json({ success: true, repley });
            })
        });
    });
});

app.listen(port, () => console.log("Listening on port " + port));
