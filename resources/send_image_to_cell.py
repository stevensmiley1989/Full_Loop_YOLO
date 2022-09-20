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

def SendMail(ImgFileName,subject,destination,default_prefix='Detected'):
	global Server,Port
	global From_Add
	global UserName, UserPassword

	
	msg = MIMEMultipart()
	msg['Subject'] = subject
	msg['From'] = From_Add
	msg['To'] = destination 
	

	if os.path.isdir(ImgFileName):
		image_dir=os.listdir(ImgFileName)
		text_words=[os.path.join(ImgFileName,w) for w in image_dir if w.find('.txt')!=-1]
		text_words=text_words[0]
		f=open(text_words,'r')
		f_read=f.readlines()
		f.close()
		text="{} at {} for {}; ".format(default_prefix,datetime.datetime.now(),os.path.basename(ImgFileName).split('.')[0])
		for line in f_read:
			line=line.replace('\n','')
			text+=line
			print(line)
		text_words=MIMEText(text)
		msg.attach(text_words)
	else:
		if default_prefix=='Detected':
			text = MIMEText("{} at {} for {}\n".format(default_prefix,datetime.datetime.now(),os.path.basename(ImgFileName).split('.')[0]))
		else:
			text=MIMEText(default_prefix)
		msg.attach(text)
	if os.path.isdir(ImgFileName)==False:
		with open(ImgFileName, 'rb') as f:
			img_data = f.read()
		image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
		msg.attach(image)
	else:
		image_dir=os.listdir(ImgFileName)
		image_list=[os.path.join(ImgFileName,w) for w in image_dir if w.find('.jpg')!=-1 or w.find('.png')!=-1]
		for img_i in image_list:
			with open(img_i,'rb') as f:
				img_data_i=f.read()
			f.close()
			image=MIMEImage(img_data_i,name=os.path.basename(img_i))
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
	parser.add_argument("--default_prefix",type=str,default="Detected",help='prefix to use for sending alerts')
	args = parser.parse_args()
	print('args.destinations',args.destinations)
	destinations=args.destinations.split(';')
	print('destinations:')
	print([w for w in destinations])
	default_prefix=args.default_prefix
	#destinations=args.destinations.split(";")
	main_message=args.main_message
	img_path=args.img_path
	for destination in destinations:
		SendMail(img_path,main_message,destination,default_prefix)
