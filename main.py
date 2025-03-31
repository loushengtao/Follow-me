import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import argparse
from utils import *
from path_attacker import Follow_me_Attacker
from video_object import video_data
def load_argparse():
    parser = argparse.ArgumentParser(description="Follow-me: Deceiving Trackers with Fabricated Paths")
    
    parser.add_argument("--mode", type=str, required=True, choices=['attack','vis'],help="Attack mode or visualize mode")
    parser.add_argument("--victim_dir", type=str, required=True, help="Path to victim dir")
    parser.add_argument("--data_type", type=str, required=True, choices=['dataset','frames','avi','mp4'],help="Victim data type")
    parser.add_argument("--cases",
        nargs='+',
        required=True,
        choices=['arb','case1','case2','case3'],
        help=(
            "Specify one or more attack cases. ")
    )
    parser.add_argument("--datait_name", default='',type=str, help="If the data type is 'dataset', provide the column names.")
    parser.add_argument("--dataset_name", default='VOT2018',type=str, help="If the data type is 'dataset', provide the dataset name.")
    parser.add_argument("--victim_root", default='testdata/',type=str, help="Root dir of victim dirs.")
    parser.add_argument("--max_len", default=80,type=int, help="Maximum number of frames to process for victim data.")
  
    return parser


def main():
    parser = load_argparse()
    args = parser.parse_args()

    if args.mode=='attack':
        if args.data_type=='frames':
            video_obj=video_data(os.path.join(args.victim_dir,'ori_frames'),args.data_type)
            path_attacker=Follow_me_Attacker(video_obj,args.victim_dir,max_length=args.max_len)
            for case in args.cases:
                if case=='arb':
                    path_attacker.arb_trajectory_attack()
                else:
                    path_attacker.cases_trajectory_attack(type=case)
        elif args.data_type=='dataset':
            video_obj=video_data(args.victim_dir,args.data_type,dataset_name=args.dataset_name,choose_seq=args.datait_name)
            path_attacker=Follow_me_Attacker(video_obj,args.victim_root,max_length=args.max_len)
            video_obj=video_data('a/VOT2018','dataset',dataset_name='VOT2018',choose_seq='rabbit')
            for case in args.cases:
                if case=='arb':
                    path_attacker.arb_trajectory_attack()
                else:
                    path_attacker.cases_trajectory_attack(type=case)
        else:
            video_obj=video_data(args.victim_dir,args.data_type)
            path_attacker=Follow_me_Attacker(video_obj,args.victim_dir,max_length=args.max_len)

            for case in args.cases:
                if case=='arb':
                    path_attacker.arb_trajectory_attack()
                else:
                    path_attacker.cases_trajectory_attack(type=case)
  
    elif args.mode=='vis':
        if args.data_type=='frames':
            video_obj=video_data(os.path.join(args.victim_dir,'ori_frames'),args.data_type)
            path_attacker=Follow_me_Attacker(video_obj,args.victim_dir,max_length=args.max_len)
            path_attacker.visiualize(vis_type_lis=args.cases)

        elif args.data_type=='dataset':
            video_obj=video_data(args.victim_dir,args.data_type,dataset_name=args.dataset_name,choose_seq=args.datait_name)
            path_attacker=Follow_me_Attacker(video_obj,args.victim_root,max_length=args.max_len)
            path_attacker.visiualize(vis_type_lis=args.cases)

        else:
            video_obj=video_data(args.victim_dir,args.data_type)
            path_attacker=Follow_me_Attacker(video_obj,args.victim_dir,max_length=args.max_len)
            path_attacker.visiualize(vis_type_lis=args.cases)


main()
    