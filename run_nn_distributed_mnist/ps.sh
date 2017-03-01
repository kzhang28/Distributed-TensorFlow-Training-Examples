python  mnist_nn_distibuted_placeholder.py \
	--ps_hosts=ec2-75-101-248-97.compute-1.amazonaws.com:2222 \
	--worker_hosts=ec2-54-210-179-194.compute-1.amazonaws.com:2224,ec2-54-210-179-194.compute-1.amazonaws.com:2226 \
	--job_name=ps \
	--task_index=0 \
	--num_workers=2
