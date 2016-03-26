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
  params['mfile']="-11"
  model_path=params["wd"]+"/cp/"+params['mfile']
  data=du.load_pose(params)
  data_train=(data[0],data[1])
  data_test=(data[2],data[3])

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-params["init_scale"],params["init_scale"])

    # m = mp.get_model(is_training=True,params=params)
    mtest = mp.get_model(is_training=False,params=params)

    saver = tf.train.Saver()

    # tf.initialize_all_variables().run()
    saver.restore(sess=session,save_path=model_path)
    test_err = run_epoch(session, mtest,tf.no_op(),params, data_test,is_training=False)
    print("Test Err: %.5f" % test_err)



if __name__ == "__main__":
  tf.app.run()