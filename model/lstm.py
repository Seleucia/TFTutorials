import tensorflow as tf

class lstm(object):

  def __init__(self,is_training,params):
    self.batch_size = batch_size = params["batch_size"]
    self.num_steps = num_steps = params["seq_length"]
    self._Y_vals=[]
    size = params['n_hidden']
    input_size = params['input_size']
    keep_prob=params['keep_prob']
    max_grad_norm=params['max_grad_norm']

    self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps,input_size])
    self._targets = tf.placeholder(tf.float32, [batch_size*num_steps,params["n_output"]])
    self._zeros=tf.zeros([batch_size*num_steps,params["n_output"]],tf.float32)

    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=size,input_size=input_size)
    if is_training and keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=keep_prob)

    lstm_cell2 = tf.nn.rnn_cell.LSTMCell(num_units=size,input_size=size)
    if is_training and keep_prob < 1:
      lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell2, output_keep_prob=keep_prob)

    lstm_cell3 = tf.nn.rnn_cell.LSTMCell(num_units=size,input_size=size)
    if is_training and keep_prob < 1:
      lstm_cell3 = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell3, output_keep_prob=keep_prob)


    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell,lstm_cell2,lstm_cell3])

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
    self._cost = cost = tf.reduce_mean(loss)
    self._final_state = state

    self._tvars = tf.trainable_variables()

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, self._tvars),max_grad_norm)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, self._tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def tvars(self):
    return self._tvars

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
