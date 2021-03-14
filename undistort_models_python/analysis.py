import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '..')
from classes.helper_functions import helper_functions
from classes.undistort_models import undistortion
from classes.roi import roi
import os
#### specify these parameters
loc="Kennedy"
folder_path="example_file"

if __name__ == '__main__':
    hel=helper_functions()
    undist=undistortion(location=loc)

    ### creating output folder
    output_folder=hel.create_output_folder(folder_path)

    numb_files=0
    ### loop in the given folder
    print("Processing files estimated number of files:",len(os.listdir(folder_path)))

    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):
            #### load the files txt:
            #### MAYBE CHANGE THIS FUNCTION
            timestamps,coordinates=hel.read_coordinate_file_w_timestamps(os.path.join(folder_path,filename))
            file_name_output= "world_" + filename
            file_output = open(os.path.join(output_folder,file_name_output), "w+")
            #### goint through all lines
            for i in range(0, coordinates.shape[0]):
                string = timestamps[i]
                #print(string)
                ### going through all detections in one line
                for j in range(0, len(coordinates[i])):
                    pixel = undist.map_to_pixel(coordinates[i][j])
                    world_point = undist.get_world_coordinate_distorted(pixel[0], pixel[1])
                    ### this is false, if outside ROI.
                    if world_point:
                        string += str(world_point[0]) + "," + str(world_point[1])
                        if j != len(coordinates[i]) - 1:
                            string += ":"
                ##DELETING LAST :
                if len(string) > 0:
                    if string[-1] == ":":
                        string = string[:-1]
                string += "\n"
                file_output.write(string)
            numb_files=numb_files+1
            print("Number of files processed:",numb_files)
            ####
        else:
            continue


def update_minute(seconds,minute,hour):
    if seconds>60:
        minute += 1
        seconds=seconds-60
    if minute>60:
        hour +=1
        minute=minute-60
    if int(hour)<10:
        hour="0"+str(hour)
    sec_str = float("{:.3f}".format(sec))
    return seconds,minute,hour