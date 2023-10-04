import smtplib
from email.mime.text import MIMEText


class EpochEmail():
    def __init__(self, email_id: str, app_password: str, to_email: str) -> None:
        self.email_id = email_id
        self.app_password = app_password
        self.to_email = to_email

        # use gmail
        self.s = smtplib.SMTP('smtp.gmail.com', 587)
        self.s.starttls()
        self.s.login(email_id, app_password)

    
    def send_messages(self, subject: str, message: str) -> None:
        msg = MIMEText(message)
        msg['Subject'] = subject

        self.s.sendmail(self.email_id, self.to_email, msg.as_string())

    # 세션 종료
    def quit_session(self) -> None:
        self.s.quit()

