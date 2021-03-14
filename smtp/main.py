# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import smtplib, ssl
#from email.MIMEMultipart import MIMEMultipart
#from email.MIMEText import MIMEText

port = 587  # For SSL
password = "1234abcd1234"
FROM="aau_byrum@outlook.dk"
SERVER= "smtp.office365.com"
# Create a secure SSL context
#context = ssl.create_default_context()

#server=smtplib.SMTP_SSL("smtp.office365.com", port)
#server.login("aau_byrum@outlook.dk", password)


server = smtplib.SMTP(SERVER, 25)
server.connect(SERVER,587)
server.ehlo()
server.starttls()
server.ehlo()
server.login(FROM, password)
sender_email = "my@gmail.com"

message = """\
Subject: Hi there

This message is sent from Python."""
#text = msg.as_string()
server.sendmail(FROM, "mapo17@student.aau.dk", message)
server.quit()

