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
  --random_dag=True \
  --train_epochs=310 >$LOG_DIR/train.$2.log 2>&1 &
