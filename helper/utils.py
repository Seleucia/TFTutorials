import numpy
import os
import datetime
import numpy as np
import pickle


def prep_pred_file(params):
    f_dir=params["wd"]+"/pred/";
    if not os.path.exists(f_dir):
            os.makedirs(f_dir)
    f_dir=params["wd"]+"/pred/"+params["model"];
    if not os.path.exists(f_dir):
            os.makedirs(f_dir)
    map( os.unlink, (os.path.join( f_dir,f) for f in os.listdir(f_dir)) )

def write_pred(est,bindex,G_list,params):
    batch_size=est.shape[0]
    seq_length=est.shape[1]
    s_index=params["batch_size"]*bindex*seq_length
    f_dir=params["wd"]+"/pred/"+params["model"]+"/"
    for b in range(batch_size):
        for s in range(seq_length):
            diff_vec=est[b][s]*2
            vec_str = ' '.join(['%.6f' % num for num in diff_vec])
            p_file=f_dir+os.path.basename(G_list[s_index])
            with open(p_file, "a") as p:
                p.write(vec_str)
            s_index+=1


def get_loss_tf(gt,est):
    batch_seq=gt.shape[0]
    loss=0
    for b in range(batch_seq):
            diff_vec=np.abs(gt[b].reshape(14,3) - est[b].reshape(14,3))*2 #13*3
            sq_m=np.sqrt(np.sum(diff_vec**2,axis=1))
            loss +=np.nanmean(sq_m)
    loss/=(batch_seq)
    return (loss)

def get_loss(gt,est):
    batch_size=gt.shape[0]
    seq_length=gt.shape[1]
    loss=0
    for b in range(batch_size):
        for s in range(seq_length):
            diff_vec=np.abs(gt[b][s].reshape(14,3) - est[b][s].reshape(14,3))*2 #13*3
            sq_m=np.sqrt(np.sum(diff_vec**2,axis=1))
            loss +=np.nanmean(sq_m)
    loss/=(seq_length*batch_size)
    return (loss)

def get_loss_bb(gt,est):
    sf="/home/coskun/PycharmProjects/RNNPose21/daya/blanket.txt"
    batch_size=gt.shape[0]
    seq_length=gt.shape[1]
    loss=0
    loss_list=[]
    seq_list=[]
    b_seq_list=[]
    with open(sf,"a") as f_handle:
        for b in range(batch_size):
            seq_los=[0]*seq_length
            for s in range(seq_length):
                diff_vec=np.abs(gt[b][s].reshape(14,3) - est[b][s].reshape(14,3))*2 #14,3
                val=np.sqrt(np.sum(diff_vec**2,axis=1))
                for i in range(14):
                    f=val[i]
                    f_handle.write("%f"%(f))
                    if(i<13):
                        f_handle.write(";")
                f_handle.write('\n')
                b_l=np.nanmean(np.sqrt(np.sum(diff_vec**2,axis=1)))
                loss_list.append(b_l)
                seq_los[s]=b_l
                loss +=np.nanmean(np.sqrt(np.sum(diff_vec**2,axis=1)))
            b_seq_list.append(seq_los)
        seq_list=np.mean(b_seq_list,axis=0)
        loss/=(seq_length*batch_size)
    return (loss,loss_list,seq_list)


def start_log(params):
    log_file=params["log_file"]
    create_file(log_file)

    ds= get_time()

    log_write("Run Id: %s"%(params['rn_id']),params)
    log_write("Deployment notes: %s"%(params['notes']),params)
    log_write("Running mode: %s"%(params['run_mode']),params)
    log_write("Running model: %s"%(params['model']),params)
    log_write("Batch size: %s"%(params['batch_size']),params)
    log_write("Sequence size: %s"%(params['seq_length']),params)

    log_write("Starting Time:%s"%(ds),params)
    log_write("size of training data:%f"%(params["len_train"]),params)
    log_write("size of test data:%f"%(params["len_test"]),params)

def get_time():
    return str(datetime.datetime.now().time()).replace(":","-").replace(".","-")

def create_file(log_file):
    log_dir= os.path.dirname(log_file)
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    if(os.path.isfile(log_file)):
        with open(log_file, "w"):
            pass
    else:
        os.mknod(log_file)

def log_to_file(str,params):
    with open(params["log_file"], "a") as log:
        log.write(str)

def log_write(str,params):
    print(str)
    ds= get_time()
    str=ds+" | "+str+"\n"
    log_to_file(str,params)

def log_read(mode,params):
    wd=params["wd"]
    filename=params['log_file']
    with open(wd+"/logs/"+filename) as file:
        data = file.read()
        lines = data.split("\n")
        i=0
        list=[]
        for line in lines:
            if mode+"-->" in line:
                epoch=0
                error=0.
                sl=line.split("|")
                for s in sl:
                    if "epoch" in s:
                        epoch=int(s.strip().split(" ")[2])
                    if "error" in s:
                        error=float(s.strip().split(" ")[1])
                list.append((epoch,error))
    #numpy.array([[epoch,error] for (epoch,error) in list_val])[:,1] #all error
    return list

def log_read_train(params):
    wd=params["wd"]
    mode="TRAIN"
    filename=params['log_file']
    with open(wd+"/logs/"+filename) as file:
        data = file.read()
        lines = data.split("\n")
        i=0

        list=[]
        for line in lines:
            if mode+"-->" in line:
                epoch=0
                batch_index=0
                error=0.
                sl=line.split("|")
                for s in sl:
                    if "epoch" in s:
                        epoch=int(s.strip().split(" ")[2])
                    if "error" in s:
                        error=float(s.strip().split(" ")[1])
                    if "minibatch" in s:
                        batch_index=int(s.strip().split(" ")[1].split("/")[0])
                list.append((epoch,batch_index,error))
    #numpy.array([[b, c, d] for (b, c, d) in list_val if b==1 ])[:,2] #first epoch all error
    return list