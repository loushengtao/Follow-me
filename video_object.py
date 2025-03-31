import cv2
import os
from toolkit.datasets import DatasetFactory
from pysot.utils.bbox import get_axis_aligned_bbox
import numpy as np
class video_data():
    def __init__(self,source_root,source_type,relabel=False,dataset_name=None,choose_seq=None):
        self.source_root=source_root
        self.source_type=source_type
        self.save_ori_path=os.path.join(self.source_root,'ori_frames')
        self.ori_frame_lis=[]
        self.adv_frame_lis=[]
        self.ori_box_lis=[]
        self.adv_box_lis=[]
        self.gt_lis=[]
        if source_type=='dataset':
            assert choose_seq is not None, 'Please select a viedo sequence among dataset.'
            assert dataset_name is not None, 'Please provide the name of dataset.'
        self.dataset_name=dataset_name
        self.choose_seq=choose_seq
        self.initialize_data()
        self.video_labeling(Relabel=relabel)
    
    def initialize_data(self):
        if self.source_type=='avi' or self.source_type=='mp4':
            cap = cv2.VideoCapture(os.path.join(self.source_root,'victim.'+self.source_type))
            if not cap.isOpened():
                print(f"Error: Couldn't open video file {self.source_root}")
                return
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.ori_frame_lis.append(frame)
            os.makedirs(self.save_ori_path,exist_ok=True)
            for idx in range(len(self.ori_frame_lis)):
                cv2.imwrite(os.path.join(self.save_ori_path,'%d.png'%idx),self.ori_frame_lis[idx])
        elif self.source_type=='frames':          
            img_lis=os.listdir(self.source_root)
            # print(img_lis[0].split('/')[-1].split('.')[0])
            images = sorted(img_lis,key=lambda x: int(x.split('/')[-1].split('.')[0]))
            for img in images:
                temp_frame=cv2.imread(os.path.join(self.source_root,img))
                self.ori_frame_lis.append(temp_frame)
        elif self.source_type=='dataset':
            dataset = DatasetFactory.create_dataset(name=self.dataset_name,
                                            dataset_root=self.source_root,
                                            load_img=False)
            for v_idx, video in enumerate(dataset):
                if video.name !=self.choose_seq:
                    continue
                for idx, (img, gt_bbox) in enumerate(video):
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    self.gt_lis.append(gt_bbox_)
                    self.ori_frame_lis.append(img)
                break
        else:
            raise AssertionError('Input source format is not supported!')
    
    def video_labeling(self,Relabel=False):
        video_name='labeling'
        if self.source_type=='dataset':
            return
        else:            # if source_type is avi/mp4 or frame, start labeling.
            if self.source_type=='frames':
                root_path=os.path.dirname(self.source_root)
            else:
                root_path=self.source_root
            # print(root_path)
            if 'gt.txt' not in os.listdir(root_path) or Relabel:
                init_frame=self.ori_frame_lis[0]
                init_rect = cv2.selectROI('Enclose the target,then press Enter.', init_frame, False, False)
                self.gt_lis.append(list(init_rect))
                cv2.destroyAllWindows()
                with open(os.path.join(root_path,'gt.txt'), 'w') as file:
                    file.write(','.join(map(str,init_rect)))
            else:
                with open(os.path.join(root_path,'gt.txt'), 'r') as file:
                    content = file.read()
                init_rect =  list(map(int, content.split(',')))
                self.gt_lis.append(list(init_rect))
                
                    
# ### test1：frames
# video_1=video_data('testdata/ants/ori_frames/','frames',relabel=True)
# print(video_1.ori_frame_lis)
# print(video_1.gt_lis[0])
# ### test2: avi/mp4
# video_2=video_data('testdata/bag/bag.avi','avi')
# #  print(video_2.ori_frame_lis)
# print(video_2.gt_lis[0])
# ### test3: dataset
# video_3=video_data('D:/SOTATTACK/pysot-master/testing_dataset/OTB100/','dataset',dataset_name='OTB100',choose_seq='MountainBike')
# # print(video_3.ori_frame_lis)
# print(video_3.gt_lis[0])