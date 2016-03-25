import time
import helper.config as cf
import helper.dt_utils as du
import helper.utils as u
import numpy as np
import tensorflow as tf

class RNNPose(object):

  def __init__(self, is_training,params):
    self.batch_size = batch_size = params["batch_size"]
    self.num_steps = num_steps = params["seq_length"]
    self._Y_vals=[]
    size = params['n_hidden']
    input_size = 1024
    keep_prob=params['keep_prob']
    max_grad_norm=params['max_grad_norm']

    self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps,input_size])
    self._targets = tf.placeholder(tf.float32, [batch_size*num_steps,params["n_output"]])
    self._zeros=tf.zeros([batch_size*num_steps,params["n_output"]],tf.float32)

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=size,input_size=input_size)
    if is_training and keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    if is_training and keep_prob < 1:
       self._input_data = tf.nn.dropout(self._input_data, keep_prob)

    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(self._input_data[:,time_step,:], state)
        outputs.append(cell_output)



    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    softmax_w = tf.get_variable("softmax_w", [size, params["n_output"]])
    softmax_b = tf.get_variable("softmax_b", [params["n_output"]])
    self._Y_vals = tf.tanh(tf.matmul(output, softmax_w) + softmax_b)
    tmp = self._Y_vals - self._targets
    tmpt=tf.select(tf.is_nan(tmp),self._zeros,tmp)
    loss=  tf.nn.l2_loss(tmpt)
    # loss = tf.nn.seq2seq.sequence_loss_by_example(
    #     [logits],
    #     [tf.reshape(self._targets, [-1])],
    #     [tf.ones([batch_size * num_steps])])
    self._cost = cost = tf.reduce_mean(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def Y_vals(self):
    return self._Y_vals


  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op



def run_epoch(session, m, eval_op,params,data,is_training=False, verbose=False):
  """Runs the model on the given data."""
  (X,Y)=data
  batch_size=params['batch_size']
  n_batches = len(X)
  n_batches /= batch_size

  nb_epochs=params['n_epochs']
  epoch_size = (len(X) // batch_size)
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  for minibatch_index in range(n_batches):
    x=X[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
    y=Y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]#60*20*54
    y=y.reshape(batch_size*params["seq_length"],params["n_output"])
    cost, state,Y_est, _ = session.run([m.cost, m.final_state,m.Y_vals,eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    if(is_training==False):
       cost=u.get_loss_tf(y,Y_est)

    costs += cost
    iters += m.num_steps

  return np.exp(costs / iters)


def main(_):

  params=cf.get_params()
  data=du.load_pose(params)
  data_train=(data[0],data[1])
  data_test=(data[2],data[3])
  # params["len_train"]=X_train.shape[0]*X_train.shape[1]
  # params["len_test"]=X_test.shape[0]*X_test.shape[1]

  best_test_err=1000000
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-params["init_scale"],
                                                params["init_scale"])
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = RNNPose(is_training=True,params=params)
      saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b

    mtest = RNNPose(is_training=False,params=params)



    tf.initialize_all_variables().run()

    for i in range(params['n_epochs']):
      lr_decay = params['lr_decay'] ** max(i - params['n_epochs'], 0.0)
      m.assign_lr(session, params['lr']* lr_decay)

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_err = run_epoch(session, m, m.train_op, params,data_train,is_training=True,verbose=True)
      print("Train Err: %.3f" % train_err)
      if(i%params['test_freq']==0):
        test_err = run_epoch(session, mtest,tf.no_op(),params, data_test,is_training=False)
        print("Test Err: %.3f" % test_err)
        if(test_err<best_test_err):
          best_test_err=test_err
          saver.save(sess=session,save_path=params["model_file"],global_step=i+1,latest_filename="model_"+str(i)+"_"+str(test_err)+".p")



if __name__ == "__main__":
  tf.app.run()