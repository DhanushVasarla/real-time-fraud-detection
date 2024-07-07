# mail.py

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(sender_email, receiver_email, password, subject, body, message_list):
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    email_body = f"{body}\n\nMessage List:\n{message_list}"
    message.attach(MIMEText(body, "plain"))

    try:
        
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender_email, password)

       
        server.sendmail(sender_email, receiver_email, message.as_string())
        print("Email sent successfully")

    except smtplib.SMTPException as e:
        print(f"Error sending email: {str(e)}")

    finally:
        try:
            server.quit()  
        except AttributeError:
            pass 