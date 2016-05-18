import os

rootdir = '/home/lzc/microhzhou/'

fout = open('microhzhou_manual_files.txt', 'w')
fout1 = open('./microhzhou_images_with_star.txt', 'w')
for dirname, _, filelist in os.walk(rootdir):
    for f in filelist:
        if f.find('_autopick') != -1:
        #print>>fout, f[0:14] + ".mrc"
            print>>fout1, f[0:f.find('_autopick')] + '.mrc'
            print>>fout, f
            

