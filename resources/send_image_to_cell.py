import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import datetime
from EMAIL_INFO import From_Add, UserName, UserPassword
Server='smtp.gmail.com'
Port=587
global destinations
destinations = ['XXXYYYZZZZ@mms.att.net']

def SendMail(ImgFileName,subject,destination):
	global Server,Port
	global From_Add
	global UserName, UserPassword
	with open(ImgFileName, 'rb') as f:
		img_data = f.read()
	
	msg = MIMEMultipart()
	msg['Subject'] = subject
	msg['From'] = From_Add
	msg['To'] = destination 
	
	text = MIMEText("This target of interest was detected at {}\n for {}\n".format(datetime.datetime.now()),ImgFileName)
	msg.attach(text)
	image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
	msg.attach(image)
	print('Sent message to {}\n'.format(destination))
	s = smtplib.SMTP(Server, Port)
	s.ehlo()
	s.starttls()
	s.ehlo()
	s.login(UserName, UserPassword)
	s.sendmail(From_Add, destination, msg.as_string())
	s.quit()
if __name__=='__main__':
	import datetime
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--destinations",type=str,default='XXXYYYZZZZ@mms.att.net',help='phone numbers to send text message updates to')
	parser.add_argument("--main_message",type=str,default='Detection',help='Message')
	parser.add_argument("--img_path",type=str,default="../dataset/sample_rc_car/JPEGImages/20220511_toyrc_DJI_0008_00000001.jpg",help="path to image chip")
	args = parser.parse_args()
	print('destinations:')
	print([w for w in destinations])
	destinations=args.destinations.split(";")
	main_message=args.main_message
	img_path=args.img_path
	SendMail(img_path,main_message,destinations[0])