# # attack frames data
# python main.py \
#     --mode attack \
#     --victim_dir testdata/ants \
#     --data_type frames \
#     --cases arb case1 case2 case3 \


# # attack dataset data
# python main.py \
#     --mode attack \
#     --victim_dir a/VOT2018 \
#     --data_type dataset \
#     --cases arb case1 case2 case3 \
#     --dataset_name VOT2018 \
#     --datait_name rabbit

# attack avi/mp4 data   ***
python main.py \
    --mode attack \
    --victim_dir testdata/bag/ \
    --data_type avi \
    --cases arb case1 case2 case3 \