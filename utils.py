import numpy as np

def perframe_target(arb_tra,start_ind,velocity):
    gap=0
    temp=0
    min_dis=10000
    end_ind=start_ind
    for i in range(start_ind,len(arb_tra)-1):
        gap+=np.sqrt((arb_tra[i][0]-arb_tra[i+1][0])**2+(arb_tra[i][1]-arb_tra[i+1][1])**2)
        #print(gap,velocity)
        temp=abs(gap-velocity)
        #print(temp)
        if temp<=min_dis:
            min_dis=temp
            end_ind=i
        else:
            end_ind=i
            break
    #print(end_ind)
    return end_ind

def mapping_process(point_list,N_frame):
    target_tra=[]
    # compute constant velocity
    sum_dis=0
    for i in range(1,len(point_list)):
        sum_dis+=np.sqrt((point_list[i][0]-point_list[i-1][0])**2+(point_list[i][1]-point_list[i-1][1])**2)
    velocity=sum_dis/N_frame
    print('constant velocity is:',velocity)
    ind=0
    #print(point_list)
    for i in range(N_frame):
        ind_new=perframe_target(point_list,ind,velocity)
        ind=ind_new
        target_tra.append(point_list[ind])
    return target_tra

def check_path(source_root):
    if not source_root.endswith('/'):
        source_root+='/'
    return source_root


def load_config(filename):
    with open(filename, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return {}