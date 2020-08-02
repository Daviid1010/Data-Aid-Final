<?php
$servername = " den1.mysql4.gear.host ";
$username = "dataaid";
$password = "Gv365NTY-!W6";
$dbname = "dataaid";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$sql = "SELECT * FROM users";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    // output data of each row
    while($row = $result->fetch_assoc()) {
        echo "id: " . $row["CustomerId"]. " - Email: " . $row["email"]. " " . $row["pword"]. "<br>";
    }
} else {
    echo "0 results";
}
$conn->close();
?>