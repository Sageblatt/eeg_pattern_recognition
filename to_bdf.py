import sys
import numpy as np
from EDFlib import edfwriter as ef
import matplotlib.pyplot as plt


sig1 = np.load('data/np_filt/IIS Channel_1_filt.npy')
sig2 = np.load('data/np_filt/IIS Channel_2_filt.npy')[1:]
sig3 = np.load('data/np_filt/IIS Channel_3_filt.npy')[1:]
sig4 = np.load('data/np_filt/IIS Channel_4_filt.npy')[1:]
sig5 = np.load('data/np_filt/IIS Channel_5_filt.npy')[1:]
sig6 = np.load('data/np_filt/IIS Channel_6_filt.npy')[1:]
s = np.load('data/np_raw/IIS Channel 1       .npy')[1:]

sf = sig1[0]  # samplerate signal 1
sf = 250  # samplerate signal 1
sig1= sig1[1:]
edfsignals = 6

sig = [sig1, sig2, sig3, sig4, sig5, sig6]

m = [np.max(si) for si in sig]
m = max(m)+1

hdl = ef.EDFwriter("output.bdf", ef.EDFwriter.EDFLIB_FILETYPE_BDFPLUS, edfsignals)

for chan in range(0, edfsignals):
  if hdl.setPhysicalMaximum(chan, m) != 0:
    print("setPhysicalMaximum() returned an error")
    sys.exit()
  if hdl.setPhysicalMinimum(chan, -m) != 0:
    print("setPhysicalMinimum() returned an error")
    sys.exit()
  if hdl.setDigitalMaximum(chan, 8388607) != 0:
    print("setDigitalMaximum() returned an error")
    sys.exit()
  if hdl.setDigitalMinimum(chan, -8388608) != 0:
    print("setDigitalMinimum() returned an error")
    sys.exit()
  if hdl.setPhysicalDimension(chan, "uV") != 0:
    print("setPhysicalDimension() returned an error")
    sys.exit()
  if hdl.setSignalLabel(chan, f"channel {chan+1}") != 0:
    print("setSignalLabel() returned an error")
    sys.exit()
  if hdl.setSampleFrequency(chan, sf) != 0:
    print("setSampleFrequency() returned an error")
    sys.exit()
# if hdl.setPreFilter(0, "HP:0.05Hz LP:250Hz N:60Hz") != 0:
#   print("setPreFilter() returned an error")
#   sys.exit()
# if hdl.setPreFilter(1, "HP:0.05Hz LP:250Hz N:60Hz") != 0:
#   print("setPreFilter() returned an error")
#   sys.exit()
# if hdl.setTransducer(0, "AgAgCl cup electrode") != 0:
#   print("setTransducer() returned an error")
#   sys.exit()
# if hdl.setTransducer(1, "AgAgCl cup electrode") != 0:
#   print("setTransducer() returned an error")
#   sys.exit()

# if hdl.setPatientCode("1234567890") != 0:
#   print("setPatientCode() returned an error")
#   sys.exit()
# if hdl.setPatientBirthDate(1913, 4, 7) != 0:
#   print("setPatientBirthDate() returned an error")
#   sys.exit()
# if hdl.setPatientName("Smith J.") != 0:
#   print("setPatientName() returned an error")
#   sys.exit()
# if hdl.setAdditionalPatientInfo("normal condition") != 0:
#   print("setAdditionalPatientInfo() returned an error")
#   sys.exit()
# if hdl.setAdministrationCode("1234567890") != 0:
#   print("setAdministrationCode() returned an error")
#   sys.exit()
# if hdl.setTechnician("Black Jack") != 0:
#   print("setTechnician() returned an error")
#   sys.exit()
# if hdl.setEquipment("recorder") != 0:
#   print("setEquipment() returned an error")
#   sys.exit()
# if hdl.setAdditionalRecordingInfo("nothing special") != 0:
#   print("setAdditionalRecordingInfo() returned an error")
#   sys.exit()
tmp = 0
err=0
# print(len(sig1))
try:
    for _ in range(int(np.ceil(len(sig1)/sf))):
        for i in range(edfsignals):
            err = hdl.writeSamples(sig[i][tmp:tmp+sf])
        tmp += sf
except:
    print('error')


# plt.plot(sig1[1:251])
# plt.plot(s[1:251], figure=plt.figure(figsize=(10.0, 8.0)))
if err != 0:
  print("writeSamples() returned error: %d" %(err))

# if not hdl.writeAnnotation(0, -1, "Recording starts"):
#   print("writeAnnotation() returned an error")
# if not hdl.writeAnnotation(40000, 20000, "Test annotation"):
#   print("writeAnnotation() returned an error")

hdl.close()


