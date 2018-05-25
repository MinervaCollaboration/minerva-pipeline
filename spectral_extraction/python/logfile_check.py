#!/usr/bin/env python

#Code to check log files and send email alerts

#Import all of the necessary packages
from __future__ import division
import os
from os.path import split
import sys
import glob
import smtplib
from email.MIMEText import MIMEText
from email.MIMEMultipart import MIMEMultipart
#import math
import time
import datetime
import numpy as np
import argparse

def get_t_statuses(fl):
    """ Returns array with good flags for each telescope """
    for line in fl:
        pass
    ### Good flags are just the last line of the file, in an array
    status = line.split(" ")
    return np.array((status))

parser = argparse.ArgumentParser()
parser.add_argument("-d","--date",help="Date to run log_file check in 'nYYYYMMDD' format (default is current day)")
parser.add_argument("-n","--dry_run",help="Run without sending emails", action="store_true")
args_in = parser.parse_args()

if args_in.date is not None:
    date = args_in.date
else:
    date = datetime.datetime.today().strftime('n%Y%m%d')

try:
    data_dir = os.environ['MINERVA_DATA_DIR']
except KeyError:
    print("Must set MINERVA_DATA_DIR")
    exit(0)
#    data_dir = "/uufs/chpc.utah.edu/common/home/bolton_data0/minerva/data"

try:
    redux_dir = os.environ['MINERVA_REDUX_DIR']
except KeyError:
    print("Must set MINERVA_REDUX_DIR")
    exit(0)
#    redux_dir = "/uufs/chpc.utah.edu/common/home/bolton_data0/minerva/redux"
    
### Get both raw and processed data from chosen date
raw_data = glob.glob(data_dir+'/{}/*.fits'.format(date))
proc_data = glob.glob(redux_dir+'/{}/*.proc.fits'.format(date))
log_files = glob.glob(redux_dir+'/{}/*.log'.format(date))

### Remove frames types will not be extracted
raw_keep = np.zeros((len(raw_data)), dtype=bool)
for i in range(len(raw_keep)):
    f = raw_data[i]
    if not 'Bias' in f and not 'Dark' in f and not 'slitFlat' in f and not 'backlight' in f and not 'autofocus' in f and not 'fiberflat' in f and not 'thar' in f and not 'test' in f and not 'FiberFlat' in f and not 'ThAr' in f:
        if split(f)[1][0:4]=='bakn':
            continue
        else:
            raw_keep[i] = True
raw_data = np.array((raw_data))[raw_keep]
processed = np.zeros((len(raw_data)))
logged = np.zeros((len(raw_data)))
exp_status = np.zeros((len(raw_data),4))
idx = 0

for frame in raw_data:
    name = split(frame)[1][:-5]
    t_statuses = np.zeros((4))
    for k in range(len(proc_data)):
        if name in proc_data[k]:
            processed[idx] = 1
            break
    for l in range(len(log_files)):
        if name in log_files[l]:
            logged[idx] = 1
            with open(log_files[l], "r") as fl:
                t_statuses = get_t_statuses(fl)
            break
    exp_status[idx] = t_statuses
    idx += 1
    
def sendmail(script, subject, recipients=['m.cornachione@utah.edu']):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = 'minerva.eprv@gmail.com'
    msg['To'] = recipients[0]
    msg.attach(MIMEText(script))
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login('minerva.eprv@gmail.com','Lali!Polt80cms')
    s.sendmail('minerva.eprv@gmail.com', recipients, msg.as_string())
    s.quit()
    
### Send report on unprocessed files
script = ""
if np.sum(processed) < len(processed):
    script += "The following fits files were not successfully processed:\n"
    for i in range(len(processed)):
        if processed[i]==1:
            continue
        else:
            script += raw_data[i] + "\n"
    script += "\n"
if np.sum(logged) < len(logged):
    script += "Log files not generated for the following files:\n"
    for i in range(len(logged)):
        if logged[i]==1:
            continue
        else:
            script += raw_data[i] + "\n"
    script += "\n"
if script != "" and not args_in.dry_run:
    sendmail(script, "MINERVA Unprocessed Data Report: {}".format(date))
    
### Send report on low signal files
all_clear = True
if np.min(exp_status) <= 1: ##Flags zero and low exposures
    all_clear = False
ignore = np.array(([1, 1, 1, 1])) ## Set 0 to ignore one or more telescopes
escript = "Warning - Low/Zero Exposures found on {}\n\n".format(date)
tzeros = np.sum(exp_status==0, axis=0)
tones = np.sum(exp_status==1, axis=0)
ttwos = np.sum(exp_status==2, axis=0)
escript += "Telescope Summaries:\n"
for l in range(4):
    escript += "  T{t}: {n}/{tot} Normal, {l}/{tot} Low, {z}/{tot} Zero\n".format(t=l+1,n=ttwos[l], l=tones[l], z=tzeros[l], tot=len(raw_data))
escript += "\nList of files with no-signal exposures:\n"
for i in range(len(logged)):
    if min(exp_status[i]) == 0:
        escript += "  " + raw_data[i] + "\n"

if not all_clear and not args_in.dry_run:
    sendmail(escript, "MINERVA Low Exposure Report: {}".format(date))