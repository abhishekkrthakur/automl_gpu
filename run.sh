SECONDS=0
echo "Dumping datasets as pickles"
echo "This makes our lives easy"
echo "Time Elapsed: ", $SECONDS
python dump_datasets.py
echo "Time Elapsed: ", $SECONDS
echo "Training Evita"
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python evita.py
echo "Time Elapsed: ", $SECONDS
echo "Training flora"
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python flora.py
echo "Time Elapsed: ", $SECONDS
echo "Training helena"
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python helena.py
echo "Time Elapsed: ", $SECONDS
echo "Training tania"
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python tania.py
echo "Time Elapsed: ", $SECONDS
echo "Training yolanda"
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python yolanda.py
echo "Time Elapsed: ", $SECONDS
echo "Training complete. All files can be found in res/"
