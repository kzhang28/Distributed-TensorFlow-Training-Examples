/Users/kuozhang/anaconda/envs/tensorflow/bin/python mnist_softmax_distibuted_placeholder.py \
	--ps_hosts=localhost:2222 \
	--worker_hosts=localhost:2224,localhost:2226 \
	--job_name=ps \
	--task_index=0 \
	--num_workers=2
