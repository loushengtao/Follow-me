import argparse
import torch
import numpy as np
import time as time
import cv2
import pickle
import os
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
import torch.autograd.variable as Variable
from pysot.utils.bbox import get_axis_aligned_bbox
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker
from pysot.utils.model_load import load_pretrain
from tqdm import tqdm

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


class Follow_me_Attacker():
    def __init__(self,video_object,victim_path,max_length=80):
        self.max_iterations=30
        self.video_obj=video_object
        if self.video_obj.source_type=='dataset':
            seq_path=os.path.join(victim_path,self.video_obj.dataset_name+'_'+self.video_obj.choose_seq)
            self.victim_path=seq_path
        else:
            self.victim_path=victim_path
        self.tracker=self.init_tracker()
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.point_list = []
        self.preserver_dic={'ori':'ori_frames/','arb':'arb_adv_frames/','case1':'case1_adv_frames/',\
                            'case2':'case2_adv_frames/','case3':'case3_adv_frames/'}
        self.vis_color_dic={'ori':(0, 255, 0),'arb_gt':(255, 0, 0),'arb':(0, 0, 255),\
                            'case1':(255,255,0),'case2':(255,0,255),'case3':(0,255,255)}
        self.max_frame_len=max_length
        self.len_frame=len(self.video_obj.ori_frame_lis) if len(self.video_obj.ori_frame_lis)<self.max_frame_len else self.max_frame_len
    
    def tracking(self,tracking_type='ori',preserve=False):
        # print(self.len_frame)
        frame_lis=[]
        prediction_boxes=[self.video_obj.gt_lis[0]]
        if tracking_type=='ori':
            frame_lis=self.video_obj.ori_frame_lis
        else:
            path=os.path.join(self.victim_path,self.preserver_dic[tracking_type])
            frame_lis.append(self.video_obj.ori_frame_lis[0])
            for idx in range(1,self.len_frame):
                img_new=cv2.imread(os.path.join(path,str(idx+1)+'.png'))
                frame_lis.append(img_new)
        for idx in range(self.len_frame):
            img=frame_lis[idx]
            if idx==0:
                it=img.copy()
                self.img=it
                if self.video_obj.source_type=='dataset':
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(self.video_obj.gt_lis[0]))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    self.tracker.init(img, gt_bbox_)
                else:
                    x,y,w,h=self.video_obj.gt_lis[0]
                    cx,cy=x+(w-1)/2,y+(h-1)/2
                    self.tracker.init(img,self.video_obj.gt_lis[0])
            else:
                outputs,_=self.tracker.track(img)
                bbox = list(map(int, outputs['bbox']))
                prediction_boxes.append([bbox[0],bbox[1],bbox[2],bbox[3]])
        if preserve:  # save path
            with open(os.path.join(self.victim_path,'%s_path.pkl'%tracking_type), 'wb') as file:
                pickle.dump(prediction_boxes, file)
        return prediction_boxes
    
    
    def draw_circle(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix,self.iy=x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                self.ix,self.iy=x,y
                cv2.circle(self.img,(self.ix,self.iy),2,(0,255,0),2) 
                self.point_list.append([self.ix,self.iy])
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
    
    def draw_line(self,pic,center,draw_circle):
        cv2.circle(pic,(int(center[0]),int(center[1])),2,(0,0,255),2)
        self.point_list.append(center)
        name="Draw the path,then press 'q'."
        cv2.namedWindow(name)
        cv2.setMouseCallback(name,draw_circle)
        while(1):
            cv2.imshow(name,pic)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') :
                break
            elif k == 27:
                break
        cv2.destroyAllWindows()
        return self.point_list

    def init_tracker(self):
        parser = argparse.ArgumentParser(description='tracking demo')
        parser.add_argument('--config', default='experiments/siamrpn_r50_l234_dwxcorr/config.yaml',type=str, help='config file')
        parser.add_argument('--snapshot', default='experiments/siamrpn_r50_l234_dwxcorr/model.pth',type=str, help='model name')
        parser.add_argument('--video_name', default='demo/bag.avi', type=str,help='videos or image files')
        parser.add_argument('--dataset',default='OTB100',help='datasets')
        args = parser.parse_args(args=[])
        cfg.merge_from_file(args.config)
        model = ModelBuilder()
        model = load_pretrain(model, args.snapshot).cuda().eval()
        tracker = build_tracker(model)
        return tracker
        
    
    def get_mask(self,img,bound,constraint):
        r, c, k = img.shape
        if any([bound[0], bound[1], bound[2], bound[3]]):
            size = (r + bound[0] + bound[1], c + bound[2] + bound[3], k)
            te_im = torch.ones(size)
            if bound[0]:
                te_im[0:bound[0], bound[2]:bound[2] + c, :] = 0
            if bound[1]:
                te_im[r + bound[0]:, bound[2]:bound[2] + c, :] = 0
            if bound[2]:
                te_im[:, 0:bound[2], :] = 0
            if bound[3]:
                te_im[:, c + bound[2]:, :] = 0 
            mask = te_im[int(constraint[0]):int(constraint[1] + 1),int(constraint[2]):int(constraint[3] + 1), :]
        else:
            te_im = torch.ones(img.shape)
            mask = te_im[int(constraint[0]):int(constraint[1] + 1),int(constraint[2]):int(constraint[3] + 1), :]
        mask = mask.permute(2, 0, 1).unsqueeze(0).cuda()
        return mask
    
    def rec_im(self,img,ip_tensor,hole_im,bound,constraint):  # recover the adv region to the ori image
        r, c, k = img.shape
        im_replace=ip_tensor.squeeze(0).cpu().detach().numpy().transpose((1,2,0)).astype(np.uint8)
        hole_im[int(constraint[0]):int(constraint[1]+1),int(constraint[2]):int(constraint[3]+1),:]=im_replace
        if any(bound):
            img_new=hole_im[bound[0]:bound[0]+r,bound[2]:bound[2]+c,:].astype(np.uint8)
        else:
            img_new=hole_im.astype(np.uint8)
        return img_new
    
    def ge_ten(self,tracker,img):  # Obtain the tracking window
        #r, c, k = img.shape
        w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
        h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
        s_z = np.sqrt(w_z * h_z)
        #scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        input_ten,hole_im,bound,constraint=tracker.get_subwindow_ten(img, tracker.center_pos,round(s_x), tracker.channel_average)
        mask=self.get_mask(img,bound,constraint)
        return input_ten,mask,hole_im,bound,constraint,s_z
    
    
    def attack_once_rg(self,tracker,img,save_path,id,target_anchor,target_shift=None,type='arb',preserve=True):  # 
        # print('----processing No.%d img----'%id)    
        input_ten,mask,hole_im,bound,constraint,s_z=self.ge_ten(tracker,img)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        
        for i in range(self.max_iterations):
            input_ten=input_ten.requires_grad_(True)
            rs_ten=tracker.rs(input_ten)
            outputs = tracker.model.track(rs_ten)
            cls,loc=outputs['cls'],outputs['loc']
            # shape change from [1,10,25,25] to [3125,2]  25*25 points corresponding to 25*25*5 anchors
            delta=loc.permute(1, 2, 3, 0).contiguous().view(4, -1)
            delta[0, :] = delta[0, :] * torch.tensor(tracker.anchors[:, 2],device='cuda:0') + torch.tensor(tracker.anchors[:, 0],device='cuda:0')
            delta[1, :] = delta[1, :] * torch.tensor(tracker.anchors[:, 3],device='cuda:0') + torch.tensor(tracker.anchors[:, 1],device='cuda:0')
            score = cls.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
            score=F.softmax(score,dim=1)
            loss_sum=torch.tensor(0,dtype=torch.float32).cuda()
            for i in range(0,3125):
                if i==target_anchor:
                    loss_sum+=-1500*torch.log(score[target_anchor][1])
                    if type=='arb' or type=='case3':
                        loss_sum+=100*((delta[:, target_anchor]/torch.tensor(scale_z,device='cuda:0'))[0]-torch.tensor(target_shift[0],device='cuda:0'))**2
                        loss_sum+=100*((delta[:, target_anchor]/torch.tensor(scale_z,device='cuda:0'))[1]-torch.tensor(target_shift[1],device='cuda:0'))**2
                else:
                    loss_sum+=-torch.log(1-score[i][1])
            # print('total_loss is:',loss_sum)
            # loss_sum.requires_grad_(True)
            loss_sum.backward(retain_graph=True)
            input_ten.data=input_ten.data-mask*input_ten.grad.data.sign()
            input_ten.data=torch.clamp(input_ten.data,min=0,max=255)
            tracker.model.zero_grad()
            if input_ten.grad is not None:
                input_ten.grad.zero_()
            # print(loss_sum)
            if loss_sum<100:
                break
        # print(score[target_anchor][1])
        img_new=self.rec_im(img,input_ten,hole_im,bound,constraint)
        if preserve:
            cv2.imwrite(os.path.join(save_path,'%d.png'%id),img_new)
        return img_new
    
    def cases_trajectory_attack(self,type='case1'):
        t_sum=0
        bias=None
        self.save_path=os.path.join(self.victim_path,self.preserver_dic[type])
        os.makedirs(self.save_path,exist_ok=True)
        print('Start attacking %d frames video'%self.len_frame)
        if type=='case3':
            if os.path.exists(os.path.join(self.victim_path,'ori_path.pkl')):
                with open(os.path.join(self.victim_path,'ori_path.pkl'), 'rb') as input:
                    ori_pd = pickle.load(input)
            else:
                self.tracking(tracking_type='ori',preserve=True)
                with open(os.path.join(self.victim_path,'ori_path.pkl'), 'rb') as input:
                    ori_pd = pickle.load(input)
                
        for idx in tqdm(range(self.len_frame)):
            img=self.video_obj.ori_frame_lis[idx].copy()
            if idx==0:   # Draw the conterfeit path at first  
                it=img.copy()
                self.img=it
                if self.video_obj.source_type=='dataset':
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(self.video_obj.gt_lis[0]))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    self.tracker.init(img, gt_bbox_)
                else:
                    x,y,w,h=self.video_obj.gt_lis[0]
                    cx,cy=x+(w-1)/2,y+(h-1)/2
                    self.tracker.init(img,self.video_obj.gt_lis[0])
                
                if type=='case3':
                    first_center=[cx,cy]
                    pre_cx,pre_cy=cx,cy
                    previous_target_point=[cx,cy]    
                
            else:
                if type=="case1":
                    target_anchor=290
                elif type=="case2":
                    tar=[314,362,310,262]
                    if idx%60<=20:
                        target_anchor=tar[0]
                    elif idx%60>20 and idx%60<=30:
                        target_anchor=tar[1]
                    elif idx%60>30 and idx%60<=50:
                        target_anchor=tar[2]
                    elif idx%60>50 and idx%60<=60:
                        target_anchor=tar[3]
                elif type=='case3':
                    cx, cy, w, h = ori_pd[idx-1]
                    target_point= [2*first_center[0]-cx,2*first_center[1]-cy]      # zxdc
                    bias=[target_point[0]-previous_target_point[0],target_point[1]-previous_target_point[1]]
                    previous_target_point=target_point
                    
                    if abs(bias[0])<=13 and abs(bias[1])<=13:
                        target_anchor=1562
                    elif abs(bias[0])<=13 and abs(bias[1])>13:
                        if bias[1]>13:
                            target_anchor=1587
                        if bias[1]<-13:
                            target_anchor=1537
                    elif abs(bias[0])>13 and abs(bias[1])<=13:
                        if bias[0]>13:
                            target_anchor=1564
                        if bias[1]<-13:
                            target_anchor=1560
                    elif abs(bias[0])>13 and abs(bias[1])>13:
                        if bias[0]>13 and bias[1]>13:
                            target_anchor=1588
                        if bias[0]>13 and bias[1]<-13:
                            target_anchor=1564
                        if bias[0]<-13 and bias[1]<-13:
                            target_anchor=1536
                        if bias[0]<-13 and bias[1]>13:
                            target_anchor=1586
                t1=time.time()    
                img_new=self.attack_once_rg(self.tracker,img,self.save_path,idx+1,target_anchor,bias,type=type)
                t2=time.time()
                t_sum+=t2-t1
                av_t=t_sum/idx
                # print('av time is',av_t)
                _,_=self.tracker.track(img_new)
                
    
    def arb_trajectory_attack(self,preserve=True):
        t_sum=0
        self.save_path=os.path.join(self.victim_path,'arb_adv_frames')
        os.makedirs(self.save_path,exist_ok=True)
        print('Start attacking %d frames video\n'%self.len_frame,'Draw the trajectory and press "q" to finish')
        for idx in tqdm(range(self.len_frame)):
            img=self.video_obj.ori_frame_lis[idx].copy()
            if idx==0:   # Draw the conterfeit path at first  
                it=img.copy()
                self.img=it
                if self.video_obj.source_type=='dataset':
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(self.video_obj.gt_lis[0]))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    self.tracker.init(img, gt_bbox_)
                else:
                    x,y,w,h=self.video_obj.gt_lis[0]
                    cx,cy=x+(w-1)/2,y+(h-1)/2
                    self.tracker.init(img,self.video_obj.gt_lis[0])
                point_list=self.draw_line(self.img,[cx,cy],self.draw_circle)
                if preserve:  # save arbitrary path
                    with open(os.path.join(self.victim_path,'arb_path.pkl'), 'wb') as file:
                        pickle.dump(point_list, file)
                target_tra=mapping_process(point_list,self.len_frame)
                for i in range(1,len(target_tra)+1):
                    cv2.circle(it, (int(target_tra[i-1][0]),int(target_tra[i-1][1])),radius=1, color=(255, 0,0), thickness=2)
                cv2.imshow('Tra points',it) 
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
            else:
                t1=time.time()
                if idx==100:
                    break
                bias=[target_tra[idx][0]-target_tra[idx-1][0],target_tra[idx][1]-target_tra[idx-1][1]]
                # print(bias)
                if abs(bias[0])<=13 and abs(bias[1])<=13:
                    target_anchor=1562
                elif abs(bias[0])<=13 and abs(bias[1])>13:
                    if bias[1]>13:
                        target_anchor=1587
                    if bias[1]<-13:
                        target_anchor=1537
                elif abs(bias[0])>13 and abs(bias[1])<=13:
                    if bias[0]>13:
                        target_anchor=1564
                    if bias[1]<-13:
                        target_anchor=1560
                elif abs(bias[0])>13 and abs(bias[1])>13:
                    if bias[0]>13 and bias[1]>13:
                        target_anchor=1588
                    if bias[0]>13 and bias[1]<-13:
                        target_anchor=1564
                    if bias[0]<-13 and bias[1]<-13:
                        target_anchor=1536
                    if bias[0]<-13 and bias[1]>13:
                        target_anchor=1586
                img_new=self.attack_once_rg(self.tracker,img,self.save_path,idx+1,target_anchor,bias)
                t2=time.time()
                t_sum+=t2-t1
                av_t=t_sum/idx
            
                _,_=self.tracker.track(img_new)
        print('Attack time is: %.2f | Average time per frame is:%.2f'%(t_sum,av_t))
    
    def visiualize(self,vis_type_lis=['case1','case2','case3','arb']):
        vis_arb_gt=False
        if 'arb' in vis_type_lis:
            vis_arb_gt=True
        
        annotations = [
                {'text': 'ori', 'color': (0, 255, 0), 'row': 0},       # Green
                {'text': 'arb_gt', 'color': (255, 0, 0), 'row': 0},    # Red
                {'text': 'arb', 'color': (0, 0, 255), 'row': 0},       # Blue
                {'text': 'case1', 'color': (255, 255, 0), 'row': 1},   # Yellow
                {'text': 'case2', 'color': (255, 0, 255), 'row': 1},   # Magenta
                {'text': 'case3', 'color': (0, 255, 255), 'row': 1}    # Cyan
            ]
            
        height, width = self.video_obj.ori_frame_lis[0].shape[:2]
        
        base_scale = min(width, height) / 1000
        font_scale = 0.5 * base_scale 
        line_length = int(80 * base_scale)
        space_between = int(120 * base_scale)
        font_thickness = max(1, int(1 * base_scale))

        start_x_offset = int(width * 0.05)      
        start_y_offset = int(height * 0.9)      
        
        line_height = int(40 * base_scale)
        text_offset = int(25 * base_scale)

        videoWriter = cv2.VideoWriter(os.path.join(self.victim_path,"vis_result.mp4"),cv2.VideoWriter_fourcc(*'mp4v'),12,(width,height))

        prediction_dic={}
        ori_prediction=self.tracking(tracking_type='ori')
        # print('####################')
        if vis_arb_gt:
            filepath = os.path.join(self.victim_path,'arb_path.pkl')
            with open(filepath, 'rb') as input:
                outdata = pickle.load(input)
        for type_id in vis_type_lis:
            prediction=self.tracking(tracking_type=type_id)
            prediction_dic[type_id]=prediction
        # print('********************')
        for idx in range(self.len_frame):
            # print(idx)
            temp_frame=self.video_obj.ori_frame_lis[idx]
            frame=temp_frame.copy()
            
            for i in range(1,idx+1):        # predic
                cv2.line(frame, (int(ori_prediction[i-1][0]+ori_prediction[i-1][2]/2),int(ori_prediction[i-1][1]+ori_prediction[i-1][3]/2)),\
                    (int(ori_prediction[i][0]+ori_prediction[i][2]/2),int(ori_prediction[i][1]+ori_prediction[i][3]/2)), color=self.vis_color_dic['ori'], thickness=4)
            bbox=ori_prediction[idx]

            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])),
                            (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),self.vis_color_dic['ori'],thickness=4)
            if vis_arb_gt:
                for i in range(len(outdata)-1):
                    cv2.line(frame,(int(outdata[i][0]),int(outdata[i][1])),(int(outdata[i+1][0]),int(outdata[i+1][1])), color=self.vis_color_dic['arb_gt'], thickness=4)
            
            for type_id in vis_type_lis:
                bbox_adv=prediction_dic[type_id][idx]
                arb_prediction=prediction_dic[type_id]
                
                cv2.rectangle(frame,(int(bbox_adv[0]), int(bbox_adv[1])),
                                (int(bbox_adv[0]+bbox_adv[2]), int(bbox_adv[1]+bbox_adv[3])),self.vis_color_dic[type_id],thickness=4)
                for i in range(1,idx+1):        # predic
                    cv2.line(frame, (int(arb_prediction[i-1][0]+arb_prediction[i-1][2]/2),int(arb_prediction[i-1][1]+arb_prediction[i-1][3]/2)),\
                        (int(arb_prediction[i][0]+arb_prediction[i][2]/2),int(arb_prediction[i][1]+arb_prediction[i][3]/2)), color=self.vis_color_dic[type_id], thickness=4)
            
            for idx, ann in enumerate(annotations):
                start_x = start_x_offset + (idx % 3) * space_between
                start_y = start_y_offset + (ann['row'] * line_height)
                
                line_end_x = min(start_x + line_length, width - 10)
                cv2.line(frame, (start_x, start_y), (line_end_x, start_y), ann['color'], 2)
                
                text_y = start_y + text_offset
                text_y = min(text_y, height - 10)
                
                cv2.putText(frame, ann['text'], (start_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, ann['color'], 
                        font_thickness, cv2.LINE_AA)
            # for idx, ann in enumerate(annotations):
            #     # Determine starting positions
            #     start_x = start_x_offset + (idx % 3) * space_between
            #     start_y = start_y_offset + (ann['row'] * 40)

            #     # Draw line
            #     line_end_x = start_x + line_length
            #     cv2.line(frame, (start_x, start_y), (line_end_x, start_y), ann['color'], 2)

            #     # Calculate text position
            #     text_y = start_y + 25
            #     cv2.putText(frame, ann['text'], (start_x, text_y), font, font_scale, ann['color'], font_thickness, cv2.LINE_AA)
            
            cv2.imshow('tes',frame)
            cv2.waitKey(90)
            videoWriter.write(frame)
        videoWriter.release()
        cv2.destroyAllWindows()


if __name__=='__main__':
    from video_object import video_data
    # video_obj=video_data('testdata/ants/ori_frames/','frames')
    # path_attacker=Follow_me_Attacker(video_obj,'testdata/ants/')
    # # path_attacker=Follow_me_Attacker(video_obj,'testdata/bag/')
    # # path_attacker.arb_trajectory_attack()
    # # path_attacker.cases_trajectory_attack(type="case1")
    # # path_attacker.cases_trajectory_attack(type="case2")
    # # path_attacker.cases_trajectory_attack(type="case3")
    # path_attacker.visiualize(vis_type_lis=['case3'])
    # # path_attacker.visiualize()


    # from video_object import video_data
    # video_obj=video_data('a/VOT2018','dataset',dataset_name='VOT2018',choose_seq='rabbit')
    # path_attacker=Follow_me_Attacker(video_obj,'testdata/')
    # # path_attacker.arb_trajectory_attack()
    # # path_attacker.cases_trajectory_attack(type="case1")
    # # path_attacker.cases_trajectory_attack(type="case2")
    # # path_attacker.cases_trajectory_attack(type="case3")
    # path_attacker.visiualize()
    # # path_attacker.visiualize(vis_type_lis=['arb'])