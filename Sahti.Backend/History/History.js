const mysql = require("mysql");
const express = require("express");
const cors = require("cors");
const geoip = require('geoip-lite');

const app = express();
const port = 8009;

app.use(express.json());
app.use(cors({ origin: "*" }));

app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
    next();
});

app.post("/", (req, res) => {
    const userData = req.body.userData;
    const newUserData = req.body.newUserData;
    const actionMade = req.body.actionMade;
    const navigatedLink = req.body.navigatedLink;
    const cfIpAddress = req.headers['cf-connecting-ip']; // Cloudflare IP
    const xRealIp = req.headers['x-real-ip']; // Real IP
    const xForwardedFor = req.headers['x-forwarded-for']; // Forwarded IP
    const ipAddress = cfIpAddress || xRealIp || xForwardedFor || req.socket.remoteAddress || ''; // Use the first available IP
    const device = req.get('User-Agent');
    const location = geoip.lookup(ipAddress);

    let resObject = { success: false };

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
        switch (actionMade) {
            case "login": Login(connection, userData, ipAddress, device, location); break;
            case "logout": Logout(connection, userData, ipAddress, device, location); break;
            case "navigate": NavLinks(connection, userData, navigatedLink, ipAddress, device, location); break;
            case "addAccount": AddAccount(connection, userData, newUserData, ipAddress, device, location); break;
            // case "updateG": UpdateGraph(connection, userData, ipAddress, device, location); break; 
        }
    });
});

app.listen(port, () => console.log("Listening on port " + port));

function Login(connection, userData, ipAddress, device, location) {
    var insertQuery = 'INSERT INTO Log(super_id, actionDescri, ip_address, device, location) VALUES(?, ?, ?, ?, ?)';
    connection.query(insertQuery, [userData.super_id, "Has logged in", ipAddress, device, location], (error) => {
        if (error) {
            console.error('Error executing SELECT query for Log: ' + error.stack);
            res.status(500).json({ success: false, error: "Error executing SELECT query for Log" });
            return;
        }
        else {console.log("Log has been updated successfuly");}
        connection.end();
    })
}
function Logout(connection, userData, ipAddress, device, location) {
    var insertQuery = 'INSERT INTO Log(super_id, actionDescri, ip_address, device, location) VALUES(?, ?, ?, ?, ?)';
    connection.query(insertQuery, [userData.super_id, "Has logged out", ipAddress, device, location], (error) => {
        if (error) {
            console.error('Error executing SELECT query for Log: ' + error.stack);
            res.status(500).json({ success: false, error: "Error executing SELECT query for Log" });
            return;
        }
        else {console.log("Log has been updated successfuly");}
        connection.end();
    })
}
function NavLinks(connection, userData, navigatedLink, ipAddress, device, location) {
    var insertQuery = 'INSERT INTO Log(super_id, actionDescri, ip_address, device, location) VALUES(?, ?, ?, ?, ?)';
    connection.query(insertQuery, [userData.super_id, "Has navigated to " + navigatedLink, ipAddress, device, location], (error) => {
        if (error) {
            console.error('Error executing SELECT query for Log: ' + error.stack);
            res.status(500).json({ success: false, error: "Error executing SELECT query for Log" });
            return;
        }
        else {console.log("Log has been updated successfuly");}
        connection.end();
    })
}
function AddAccount(connection, userData, newUserData, ipAddress, device, location) {
    var insertQuery = 'INSERT INTO Log(super_id, actionDescri, ip_address, device, location) VALUES(?, ?, ?, ?, ?)';
    connection.query(insertQuery, [userData.super_id, "Has added the account with the username: " + newUserData.username, ipAddress, device, location], (error) => {
        if (error) {
            console.error('Error executing SELECT query for Log: ' + error.stack);
            res.status(500).json({ success: false, error: "Error executing SELECT query for Log" });
            return;
        }
        else {console.log("Log has been updated successfuly");}
        connection.end();
    })
}
// function UpdateGraph(connection, userData, ipAddress, device, location) {
//     connection.query('INSERT INTO Log(super_id, actionDescri, ip_address, device, location) VALUES(?, ?, ?, ?, ?)', [userData.super_id, "Has updated the charts settings", ipAddress, device, location], (error) => {
//         if (error) {
//             console.error('Error executing SELECT query for Log: ' + error.stack);
//             res.status(500).json({ success: false, error: "Error executing SELECT query for Log" });
//             return;
//         }
//         else {console.log("Log has been updated successfuly");}
//         connection.end();
//     })
// }
//  More