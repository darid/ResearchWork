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

    return im_data[0:(75*80)],im_lable

# importData("/home/darid/ResearchWork/DataCollection/Data/Data4Fallfront1.txt")
