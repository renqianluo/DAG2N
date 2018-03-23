echo Using CUDA $1, train model $2

export  CUDA_VISIBLE_DEVICES=$1
#MODEL_DIR=/hdfs/sdrgvc/v-renluo/DAG2N/model/model_$2
MODEL_DIR=model/model_$2
LOG_DIR=log
mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR
nohup python main.py \
  --data_dir=data \
  --model_dir=$MODEL_DIR \
  --train_epochs=600 \
  --N=6 \
  --num_nodes=7 \
  --batch_size=128 \
  --epochs_per_eval=5 \
  --l_max=0.2 \
  --l_min=0.0 \
  --T_0=600 \
  --random_sample=True \
  --lr_schedule=cosine >$LOG_DIR/train.$2.log 2>&1 &
