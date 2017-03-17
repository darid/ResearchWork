import os
import re
import numpy as np

work_dir = "/home/darid/ResearchWork/DataCollection/Data/"

# num = 0
# for cur_file in os.listdir(work_dir):
#     print cur_file
#     num = num + 1
#     print num

def importData(filename):


    im_lable = []
    lable_dic = {'grasp\n':np.array([0,0,0,0,0,0,0,0]),
                 'lay\n':np.array([0,0,0,0,0,0,1,0]),
                 'sit\n':np.array([0,0,0,0,0,1,0,0]),
                 'walk\n':np.array([0,0,0,0,1,0,0,0]),
                 'back\n':np.array([0,0,0,1,0,0,0,0]),
                 'EndUpSit\n':np.array([0,0,1,0,0,0,0,0]),
                 'side\n':np.array([0,1,0,0,0,0,0,0]),
                 'front\n':np.array([1,0,0,0,0,0,0,0])}

    im_data = []
    ac_data = []
    with open(filename, "r") as i_data:
        temp_data = i_data.readlines()
        im_lable.append(lable_dic[temp_data[0]])
        for i in range(len(temp_data)-1):

            pattern = re.compile("^Frame" )
            match = pattern.match(temp_data[i+1])
            if not match:
                point_data = temp_data[i+1].split("\t\t")
                for n in range(3):
                    im_data.append(float(point_data[n]))

    # print im_data[0:(75*80)],im_lable,len(im_data)

    # return im_data[0:(75*80)],im_lable
    for n in range(0,len(im_data)/75-2):
        v1 = im_data[n*75:(n+1)*75]
        v2 = im_data[(n+1)*75:(n+2)*75]
        v = map(lambda x, y: (x - y)**2, v2, v1) #calculate the speed of skeleton by Tao
        # print v,"\n", len(v),"\n"
        ###
        for m in range(0,len(v)/3):
            v_d = v[m*3]+v[m*3+1]+v[m*3+2]
            ac_data.append(v_d)

        # print len(ac_data)

    # print len(ac_data), 182*25
    return ac_data[0:(25*60)],im_lable
        ###

        # ac_data.extend(v)
    # return ac_data
    # return ac_data[0:(75*60)],im_lable


# print(importData("/home/darid/ResearchWork/DataCollection/2.txt"))

