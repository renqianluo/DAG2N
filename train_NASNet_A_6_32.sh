MODEL=NASNet_A_6_32
echo Using CUDA $1, train model $MODEL

export  CUDA_VISIBLE_DEVICES=$1
MODEL_DIR=model/model_$MODEL
LOG_DIR=log
mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR
nohup python main_es.py \
  --data_dir=data \
  --model_dir=$MODEL_DIR \
  --train_epochs=600 \
  --N=6 \
  --filters=32 \
  --num_nodes=7 \
  --drop_path_keep_prob=0.6 \
  --batch_size=32 \
  --epochs_per_eval=10 \
  --lr_max=0.025 \
  --lr_min=0.0 \
  --T_0=600 \
  --dag='NASNet_A' \
  --use_aux_head \
  --lr_schedule=cosine >>$LOG_DIR/train.$MODEL.log 2>&1 &
