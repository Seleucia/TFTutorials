import time
import helper.config as cf
import helper.dt_utils as du
import helper.utils as u
import model.model_provider as mp
import tensorflow as tf

def run_epoch(session, m, eval_op,params,data,is_training=False, verbose=False):
  """Runs the model on the given data."""
  (X,Y)=data
  batch_size=params['batch_size']
  n_batches = len(X)
  n_batches /= batch_size

  costs = 0.0
  state = m.initial_state.eval()
  for minibatch_index in range(n_batches):
    x=X[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
    y=Y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
    y=y.reshape(batch_size*params["seq_length"],params["n_output"])
    cost, state,Y_est, _ = session.run([m.cost, m.final_state,m.Y_vals,eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    if(is_training==False):
       cost=u.get_loss_tf(y,Y_est)

    costs += cost
  return costs / n_batches

def main(_):
  params=cf.get_params()
  data=du.load_pose(params)
  data_train=(data[0],data[1])
  data_test=(data[2],data[3])

  best_test_err=1000000
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-params["init_scale"],
                                                params["init_scale"])
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = mp.get_model(is_training=True,params=params)

    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mtest = mp.get_model(is_training=False,params=params)

    saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
    tf.initialize_all_variables().run()

    if(params['run_mode']>1):#mode one for training from zero
      params['mfile']="-31"
      model_path=params["wd"]+"/cp/"+params['mfile']
      saver.restore(sess=session,save_path=model_path)
    if(params['run_mode']==3):#prediction mode
      test_err = run_epoch(session, mtest,tf.no_op(),params, data_test,is_training=False)
      print("Test Err: %.5f" % test_err)
    else:
      for i in range(params['n_epochs']):
        lr_decay = params['lr_decay'] ** max(i - params['n_epochs'], 0.0)
        m.assign_lr(session, params['lr']* lr_decay)
        if params['shufle_data']==1:
           data_train=du.shuffle_in_unison_inplace(data_train[0],data_train[1])
        train_err = run_epoch(session, m, m.train_op, params,data_train,is_training=True,verbose=True)
        print("Epoch: %d, Train Err: %.5f, Learning rate: %.7f" % (i + 1,train_err, session.run(m.lr)))
        if(i%params['test_freq']==0):
          test_err = run_epoch(session, mtest,tf.no_op(),params, data_test,is_training=False)
          print("Test Err: %.5f" % test_err)
          if(test_err<best_test_err):
            best_test_err=test_err
            saver.save(sess=session,save_path=params["model_file"],global_step=i+1)

if __name__ == "__main__":
  tf.app.run()