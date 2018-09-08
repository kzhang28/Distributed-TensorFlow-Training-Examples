"""This script is the distributed version of mnist_softmax.py example provided by tensorflow
    example. The origianl non-distributed version can be found from the link:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time
#FLAGS = None
HOME_DIR="/Users/kuozhang/PycharmProjects/TensorflowSpring2017Experiment"
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string("data_dir",
                           HOME_DIR+"/temp/mnist_input_data",
                           "the dir to store mnist training data")
tf.app.flags.DEFINE_string("log_dir",HOME_DIR+"/temp/train_logs","log directory")
tf.app.flags.DEFINE_integer("total_step", 1000000, "total training step(while loop upper bound)")
tf.app.flags.DEFINE_integer("batch_size", 500, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "learning rate of SGD")
#tf.app.flags.DEFINE_float("decay_rate", 0.99, "learning rate of SGD")
#tf.app.flags.DEFINE_integer("decay_every", 10, "trigger decay every decay_every")
tf.app.flags.DEFINE_boolean("sync_replicas",True, "whether to use synchronious replicas")
tf.app.flags.DEFINE_integer("replicas_to_aggregate",None,
                            "Number of replicas to aggregate before parameter update"
                            "is applied (For sync_replicas mode only; default: "
                            "num_workers)")
tf.app.flags.DEFINE_integer("num_workers",2,
                            "Total number of workers (must be >= 1)")

FLAGS = tf.app.flags.FLAGS
IMAGE_PIXELS = 28
NUM_CLASS = 10
def main(_):
  ps_host = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"ps" : ps_host, "worker" : worker_hosts})
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
      server.join();
  elif FLAGS.job_name == "worker":
      is_chief = (FLAGS.task_index == 0)
      if FLAGS.sync_replicas:
          if FLAGS.replicas_to_aggregate is None:
              replicas_to_aggregate = FLAGS.num_workers
          else:
              replicas_to_aggregate = FLAGS.replicas_to_aggregate
      with tf.device(tf.train.replica_device_setter(
          worker_device="/job:worker/task:%d" %FLAGS.task_index,
          cluster=cluster)):

          x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS*IMAGE_PIXELS])

          W = tf.Variable(tf.zeros([IMAGE_PIXELS*IMAGE_PIXELS, NUM_CLASS]))
          b = tf.Variable(tf.zeros([NUM_CLASS]))
          y = tf.matmul(x, W) + b
          y_ = tf.placeholder(tf.float32, [None, NUM_CLASS])
          cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


          global_step = tf.Variable(0)
          lr = FLAGS.learning_rate
          optimizer = tf.train.GradientDescentOptimizer(lr)
          if FLAGS.sync_replicas:
              optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                   replicas_to_aggregate = replicas_to_aggregate,
                                                   total_num_replicas = FLAGS.num_workers,
                                                   replica_id=FLAGS.task_index,
                                                   use_locking=True,
                                                   name="Softmax_MNIST")
          train_step = optimizer.minimize(cross_entropy,global_step=global_step)
          if is_chief and FLAGS.sync_replicas:
              chief_queue_runner = optimizer.get_chief_queue_runner()
          init_op = tf.initialize_all_variables()
      sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                               logdir=FLAGS.log_dir,
                               init_op=init_op,
                               summary_op=None,
                               saver=None,
                               global_step=global_step,
                               save_model_secs=100000)
      mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
      with sv.managed_session(server.target) as sess:
          if FLAGS.sync_replicas and is_chief:
              sv.start_queue_runners(sess,[chief_queue_runner])
          step=0
          print("[INFO] Training begins!")
          begin_time = time.time()
          while not sv.should_stop() and step <FLAGS.total_step:
              #print(step)
              train_batch_xs,train_batch_ys =mnist.train.next_batch(FLAGS.batch_size)
              train_feed = {x:train_batch_xs, y_:train_batch_ys}
              _,step =sess.run([train_step,global_step],feed_dict=train_feed)
              if step % 20 == 0:
                  test_batch_xs,test_batch_ys=mnist.test.next_batch(FLAGS.batch_size)
                  test_feed = {x:test_batch_xs,y_:test_batch_ys}
                  cross_entropy_eval=sess.run(cross_entropy,feed_dict=test_feed)
                  current_time = time.time()
                  elapse_time = current_time - begin_time
                  print("Worker:%d global step:%d, cross_entropy:%f time_elapsed:%f"
                        %(FLAGS.task_index,step,cross_entropy_eval,elapse_time))
          sv.stop()
if __name__ == "__main__":
    tf.app.run()
