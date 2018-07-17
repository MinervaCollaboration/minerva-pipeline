#!/usr/bin/env python

'''
# Code to check log files and send email alerts
# Looks for:
   1. Any exposures that are not processed by the raw reduction pipeline
   2. Any telescopes that have low signals (indicating poor throughput for some reason)
   3. Any telescopes whose signal drops relative to past exposures (of the same star)
   4. Gross trace shift of > 1 pixel
   
# Will send a report to the raw reduction administrator for everything
# Will copy other addresses for low exposures (#2)
# Emails are sent from minerva.eprv@gmail.com
'''

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

def get_num_lines(fl):
    lcnt = 0
    for line in fl:
        lcnt+=1
    return lcnt

def get_t_shift(fl, line=5):
    lcnt = 0
    for line in fl:
        if (lcnt == line):
            break
        lcnt += 1
        pass
    shift = bool(line=='True')
    return shift

def get_t_ignore(fl, line=8):
    lcnt = 0
    for line in fl:
        if (lcnt == line):
            break
        lcnt += 1
        pass
    ignore = line.split(" ")
    ignore = np.array((ignore))
    ignore = ignore.astype(int)
    return ignore
    
def get_t_statuses(fl):
    """ Returns array with good flags for each telescope """
    for line in fl:
        pass
    ### Good flags are just the last line of the file, in an array
    status = line.split(" ")
    return np.array((status))

def get_t_medians(fl, lines=22):
    """ Returns array with good flags for each telescope """
    lcnt = 0
    for line in fl:
        if (lcnt == lines-2):
            break
        lcnt += 1
        pass
    ### Good flags are just the last line of the file, in an array
    medians = line.split(" ")
    m_arr = np.array((medians))
    if len(m_arr) != 4:
        print "ERROR: Problem checking medians from log file!"
        print m_arr
        exit(0)
    return m_arr
    
def get_star(fl):
    """ assumes fl is a full path with file as the last part
    """
    fl_name = os.path.split(fl)[1]
    star_plus = fl_name[10:]
    star = star_plus[0:star_plus.find('.')]
    return star

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
    
##############Input Arguments###########################3

parser = argparse.ArgumentParser()
parser.add_argument("-d","--date",help="Date to run log_file check in 'nYYYYMMDD' format (default is current day)")
parser.add_argument("-r","--reduction_admin",help="email address of raw reduction administrator", default="m.cornachione@utah.edu")
parser.add_argument("-o","--other_recipients",help="email address(es) to copy on low exposure report (may be a list, e.g. 'u1@school1.edu u2@school2.edu' etc.", default="jason.eastman@cfa.harvard.edu")
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
ignore = np.zeros((len(raw_data),4), dtype=bool)
tshift = np.zeros((len(raw_data)))
idx = 0

for frame in raw_data:
    name = split(frame)[1][:-5]
    t_statuses = np.zeros((4))
    _ignore = np.zeros((4), dtype=bool)
    for k in range(len(proc_data)):
        if name in proc_data[k]:
            processed[idx] = 1
            break
    for l in range(len(log_files)):
        if name in log_files[l]:
            logged[idx] = 1
            with open(log_files[l], "r") as fl:
                t_statuses = get_t_statuses(fl)
            with open(log_files[l], "r") as fl:
                _ignore = get_t_ignore(fl)
            with open(log_files[l], "r") as fl:
                tshift[idx] = get_t_shift(fl)
            break
    exp_status[idx] = t_statuses
    ignore[idx] = _ignore
    idx += 1
    

    
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
    
srecipients = [args_in.reduction_admin]
if script != "" and not args_in.dry_run:
    sendmail(script, "MINERVA Unprocessed Data Report: {}".format(date), recipients=srecipients)
    
### Send report on low signal files
all_clear = True
if len(raw_data) == 0:
    print "No data on {}".format(date)
    exit(0)
elif np.min(exp_status) <= 1: ##Flags zero and low exposures
    all_clear = False

escript = "Warning - Low/Zero Exposures found on {}\n\n".format(date)
tzeros = np.sum((exp_status==0)*(ignore!=0), axis=0)
tones = np.sum((exp_status==1)*(ignore!=0), axis=0)
ttwos = np.sum((exp_status==2)*(ignore!=0), axis=0)
escript += "Telescope Summaries:\n"
for l in range(4):
    escript += "  T{t}: {n}/{tot} Normal, {l}/{tot} Low, {z}/{tot} Zero\n".format(t=l+1,n=ttwos[l], l=tones[l], z=tzeros[l], tot=np.sum(ignore[:,l]))
escript += "\nList of stars with no-signal exposures:\n"
for i in range(len(logged)):
    if min(exp_status[i]) == 0:
        star = get_star(raw_data[i])
        escript += "  " + star + "\n"

other_recipients = args_in.other_recipients.split(" ")
erecipients = [args_in.reduction_admin] + other_recipients
if not all_clear and not args_in.dry_run:
    sendmail(escript, "MINERVA Low Exposure Report: {}".format(date), recipients=erecipients)
    
### Check past median values to flag a gross change in throughput
    
tpscript = "Check on throughputs for {}\n\n".format(date)
Tarr = np.array((['T1', 'T2', 'T3', 'T4']))
low_cnt = 0
for raw in raw_data:
    star = get_star(raw)
    ### Compare to the last few days
    days_to_check = 7
    past_logs = []
    today = datetime.datetime(*(time.strptime(date, 'n%Y%m%d')[0:6]))
    for i in range(days_to_check):
        days_back = datetime.timedelta(i+1)
        dnew = (today - days_back).strftime('n%Y%m%d')
        past_logs += glob.glob(os.path.join(redux_dir, dnew, '*{}*.log'.format(star)))
    if len(past_logs) == 0:
        print "No log files in last {} days for star {}".format(days_to_check, star)
    else:
        m_arr = np.zeros((len(past_logs),4))
        idxm = 0
        for log in past_logs:
            with open(log, "r") as fl:
                num_lines = get_num_lines(fl)
            with open(log) as lf:
                m_arr[idxm] = get_t_medians(lf, lines=num_lines)
            idxm += 1
        m_arr = np.median(m_arr, axis=0) #Take median across available files
        rlog = raw[0:-4] + 'log'
        rlog = rlog.replace("/data/","/redux/")
        try:
            with open(rlog, "r") as fl:
                num_lines = get_num_lines(fl)
            with open(rlog) as rd:
                m_arr_new = get_t_medians(rd, lines=num_lines)
            low_cut_pcnt = 0.5 ### Flag if new is below 50% of old median - can change this threshold as desired
            good = np.ones((4), dtype=bool)
            for k in range(4):
                if m_arr_new[k] < low_cut_pcnt*m_arr[k]:
                    good[k] = False
            if np.sum(good) < 4:
                Tlow = Tarr[good==False]
                tpscript += "File {} low on telescopes {}\n".format(raw, Tlow)
                low_cnt += 1
        except:
            print "Log file {} does not exist".format(rlog)

tprecipients = [args_in.reduction_admin]
if low_cnt > 0:
    sendmail(tpscript, "MINERVA Past Throughput Check: {}".format(date), recipients=tprecipients)
    
### Check for trace shifts
tsscript = "WARNING - trace shifts greater that 1px (relative to reference\n"
tsscript += "fiber flats (n20161123)) detected in the following exposures:\n"
ts_idx = 0
for raw in raw_data:
    if tshift[ts_idx]:
        tsscript += "  {}\n".format(raw)
    ts_idx += 1
    
tsrecipients = [args_in.reduction_admin]
if np.sum(tshift) > 0:
    sendmail(tsscript, "Trace shift found in MINERVA data on {}".format(date), recipients=tsrecipients)