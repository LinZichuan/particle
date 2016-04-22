import os

rootdir = '/home/linzichuan/Study/senior_second/particle/cryoEM-data/spliceosome-lowpass/'

fout = open('./spliceosome_manual_files.txt', 'w')
fout1 = open('./spliceosome_images_with_star.txt', 'w')
for dirname, _, filelist in os.walk(rootdir):
    for f in filelist:
        if f.find('manual') != -1:
        #print>>fout, f[0:14] + ".mrc"
            print>>fout1, f[0:f.find('manual')-1] + '.mrc'
            print>>fout, f
            

