/Users/kuozhang/anaconda/envs/tensorflow/bin/python  mnist_2hiddenLayerNN_distributed_ph.py \
	--ps_hosts=localhost:2222 \
	--worker_hosts=localhost:2224,localhost:2226 \
	--job_name=worker \
	--task_index=0 \
	--num_workers=2
