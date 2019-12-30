import argparse
import os
from yaml import load, dump, FullLoader
import sys
from shutil import copyfile, copytree
from tqdm import tqdm


ROOT = os.path.abspath("..") # get dir of FleckDetect
WORKSPACE = os.path.join(ROOT,"trainedModels/default")
WORKDIR = os.path.join(ROOT, "scripts")
d_config = {}
w_config = {}

sys.path.append(ROOT)
# Todo: Replace by access of packages, only append root
# sys.path.append(os.path.join(ROOT, "scripts/eval"))
# sys.path.append(os.path.join(ROOT, "scripts/eval/plots"))
# sys.path.append(os.path.join(ROOT, "scripts/segment"))
# sys.path.append(os.path.join(ROOT, "script/mode"))


def parser():
    ''' Guess what, it parses the program flags.
    Use -h or --help to print the help text.
    '''
    global d_config
    parser_ = argparse.ArgumentParser( \
        description='Controller Program to evaluate a trained model.', \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # TODO: Remove unused arguments
    parser_.add_argument("-w", "--workspace", type=str, default=None, help="Choose workspace directory.")
    parser_.add_argument('-l', '--list-workspaces', action='store_true',help='Evaluate models')
    parser_.add_argument('--prep-train', type=str, default=None ,help='Give name for logs and modelprefix.') 
    parser_.add_argument("--create-link-file", type=str, default=[], help="<Name of linkfile> <path/to/pred> <path/to/gt>\npaths can be relative, Dataset needs to be placed in Datasets dir!", nargs='+')
    parser_.add_argument('-e', '--evaluate', action='store_true',help='Evaluate models and plot train and validation curve')
    parser_.add_argument("--sel", type=int, help="Interation of the stored model which will be moved to currentBest dir.", default=-1)
    parser_.add_argument("--seg", type=str, default=None, help="Choose set to segment (soc|fleck|msrab).\n You can define your own test set by adding path to the workspace.")
    parser_.add_argument('--seg-all', action='store_true',help='Segment all avaliable datasets. Either val or trainset.')
    parser_.add_argument("--cal-metrics", type=str, default=None, help='Calculate Metrics for specific image set. The set needs to be defined in the workspace.\n You must need the same argument as with --segment.\n In fact it is expected to run --segment before calculating the metrics!')
    parser_.add_argument('--cal-all-metrics', action='store_true',help='Calculate metrics on all avaliable datasets. Either val or trainset.')
    parser_.add_argument('-t', '--train-mode', action='store_true',help='When accessing dataset and this flag ist set, the train set will be used, otherwise the validation set.')
    parser_.add_argument("--retrain", type=str, default=None, help='Select a dataset on which the model will be trained. The retrain option adapts the trainer and links the current workspace model to it.')
    parser_.add_argument('--plot-models', action='store_true',help='Generate one plot (PR and ROC) for all model and segmented dataset that will be found in the workspace. ')
    parser_.add_argument('-p', '--plot', type=str, default=None ,help='Generate all plots for the workspace.')    
    parser_.add_argument('--col-plots', action='store_true',help='Collect all plots.')
    parser_.add_argument('--gen-all-plots', action='store_true',help='Generate plots on all avaliable datasets. Either val or trainset.')
    parser_.add_argument('--gen-pr-roc-plot', action='store_true',help='Generate combined roc and pr plot')

    args = parser_.parse_args()

    with open("datasets.yml", "r") as f:
        d_config = load(f, Loader=FullLoader) 
        # print(config)

    create_workspace(switch_to=args.workspace)
    os.chdir(ROOT)
    return args.evaluate, args.sel, args.seg, args.cal_metrics, args.create_link_file, args.plot, args.prep_train, args.train_mode, args.seg_all, args.cal_all_metrics, args.gen_all_plots, args.list_workspaces, args.col_plots, args.gen_pr_roc_plot, args.retrain, args.plot_models
    # return config


def create_workspace(switch_to=None):
    '''Read workspace from workspace.yml.
    The idea is to switch the workspace at any time and resume again there.
    Thus the content of workspace.yml will be overwitten with the new workspace and possibly other attribute.
    '''
    global WORKSPACE
    global w_config
    with open("workspace.yml", "r") as f:
        space = load(f, Loader=FullLoader)
    
    
    # Be aware that when switching to new workspace the order of dataset and path in workspace.yml may change but is not lost!
    if switch_to:
        space["w_dir"]["current_space"] = switch_to
        with open("workspace.yml", "w") as f:
            dump(space, f)
    w_config = space
    if not os.path.exists(os.path.join(ROOT, w_config["dir"]["w_spaces"])):
        os.makedirs(os.path.join(ROOT, w_config["dir"]["w_spaces"]))
    names, space_dir = get_workspaces()
    if space["w_dir"]["current_space"] not in names + [switch_to]:
        print("No Workspace selected. Create a new workspace with:\npython controller.py -w w_space_name.")
        print("Existing Workspaces: "+ ", ".join(names) if len(names) > 0 else "")
        exit(0)


    workspace = os.path.join(ROOT, space["dir"]["w_spaces"], space["w_dir"]["current_space"])
    print("Workspace currently set to: {}".format(workspace.split("/")[-1]))
    
    if not os.path.exists(workspace):
        print("Create Workspace")
        os.makedirs(workspace)
    WORKSPACE = workspace
    try:
        print("Active model: ", load_config()[w_config["w_conf"]["sel_model"]])
    except Exception:
        print("Active model not set!")

def get_workspace_prefix():
    '''
    Return the value of the attribute "model_prefix" in the workspace specific config.yml. 
    This can be used to access workspace specific models or log files.
    Note:
    The attribute model_prefix will only be availiable after preparing a training or retraining.
    '''
    w_config_path = os.path.join(WORKSPACE,w_config["w_dir"]["current_space"]+".yml")
    if os.path.exists(w_config_path):
        with open(w_config_path, "r") as f:
            return load(f, Loader=FullLoader)[w_config["w_conf"]["model_prefix"]]
            
    else:
        raise Exception("Prefix config in {} not found!".format(w_config_path))

def get_logfile():
    '''Return path to last created log'''
    logdir = os.path.join(ROOT, w_config["dir"]["logs"])
    if not os.path.exists(logdir):
        raise Exception("Could not Find logfiles!\nDid you train a model?")
    prefix = get_workspace_prefix()
    log_name = prefix + ".out"
    return os.path.join(logdir, log_name)


def get_best_model(vis=False):
    '''Read the last logfile in model/logs and extract the val and train scores.
    Both will be stored as csv in the WORKSPACE.
    Subsequent the best train and val model will be printed as well as a plot of both scores to manually decide which model to choose.
    '''
    logfile = get_logfile()
    print(logfile)
    if not os.path.exists(logfile):
        raise Exception("Could not find logfile: {}".format(logfile))
    train_file = os.path.join(WORKSPACE, w_config["files"]["train_out"])
    val_file = os.path.join(WORKSPACE, w_config["files"]["val_out"])
    parse_log(logfile, train_out=train_file, val_out=val_file)
    det_best_model([train_file,val_file], vis=vis)


def copy_model(iteration):
    ''' When best model is selected, give the iteration of that specific model.
    If it exists, the model will be copied in the WORKSPACE.
    In the workspace specific config file is the entry sel_model added which is set to the model iteration and allows 
    the useage of multiple models in one workspace.'''
    suf_caff = ".caffemodel"
    suf_solv = ".solverstate"
    name = get_workspace_prefix()
    src_dir = w_config["dir"]["snapshot"]
    # Create model names
    modelname = "ras_" + name + "_iter_" + str(iteration) + suf_caff
    dst_model_path =  os.path.join(WORKSPACE, modelname)
    # modelstate = "ras_" + name + "_iter_" + str(iteration) + suf_solv
    # Check if model exists
    model_src_path = os.path.join(src_dir, modelname)
    if not (os.path.exists(model_src_path)):
        #  or os.path.exists(os.path.join(src_dir,modelstate))):
         raise Exception("No model found!")
    # copy if model does not exist in workspace
    if not os.path.exists(dst_model_path):
        copyfile(model_src_path, dst_model_path)
        # copyfile(model_path, os.path.join(WORKSPACE, modelstate))
    store_in_config({w_config["w_conf"]["sel_model"]: str(iteration)})
    print("New active model set to:", str(iteration))

def get_workspace_model():
    '''Return path to caffemodel according to sel_model in workspace specific config file.
    If no model specified return path to model in workspace that is found first.'''
    files = os.listdir(WORKSPACE)
    try:
        sel_model = load_config()[w_config["w_conf"]["sel_model"]]
        for name in files:
            if ("caffemodel" in name) and (str(sel_model) in name):
                return (os.path.join(WORKSPACE, name))
    except Exception:
        print("No model specified, searching for local model ...")
        for name in files:
            if ("caffemodel" in name):
                print("Found: {}".format(name))
                print("New active model set to: {}".format(name[:-11]))
                store_in_config({w_config["w_conf"]["sel_model"]: name[:-11]})
                return (os.path.join(WORKSPACE, name))
        # print("No local model found!")
    raise Exception("Cound not find a model!")

def split_path_at(key, path):
    '''Cut the part of a path before the key.'''
    splitp = path.split("/")
    try:
        return "/".join([splitp[i] for i in range(splitp.index(key), len(splitp))])
    except ValueError: 
        raise Exception("Could not create relative path to dataset.\nDid you place your images in Datasets/ directory?")

def create_link_file(img_dir, gt_dir, out_path):
    ''' Duplicated code: calMetrics for merging pred und gt in one file.
    One specialty is needed when saving the paths to a file. It needs to be a relative path, seen from Datasets.
    This is, every image name must start with Dataset/path/to/image'''
    imgs = os.listdir(img_dir)
    # print(imgs)
    gts = os.listdir(gt_dir)
    sample_count = 0
    # if len(imgs) == 0:
    #     raise Exception("Directories does not match!")
    pairs = []
    img_rel = split_path_at(w_config["dir"]["datasets"], img_dir)
    gt_rel = split_path_at(w_config["dir"]["datasets"], gt_dir)
    print("Create pairs")
    for p in tqdm(imgs): 
        p_tmp = p.split(".")[0]
        for gt in gts:
            if p_tmp == gt.split(".")[0]:

                 pairs.append((os.path.join(img_rel, p), os.path.join(gt_rel, gt)))
                 
                 break
    if len(pairs) == 0:
        raise Exception("Check your paths!\nDo both directories contain images with the same names?")
    with open(out_path, "w+") as out_file:
        print("Write pairs to file:")
        for img, gt in tqdm(pairs):
            out_file.write("{} {}\n".format(img, gt))
            sample_count += 1
        print("done!")
        return sample_count

def store_in_config(dic_var):
    '''Write workspace specific config. For now it is used to store the model prefix'''
    config_path = os.path.join(WORKSPACE, w_config["w_dir"]["current_space"]+".yml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            conf = load(f, Loader=FullLoader)
        conf.update(dic_var)

        with open(config_path, "w") as f:
            dump(conf, f)
    else:
        with open(config_path, "w+") as f:
            dump(dic_var, f)

def load_config():
    '''Return the workspace specific configuration file as dictionary.'''
    config_path = os.path.join(WORKSPACE, w_config["w_dir"]["current_space"]+".yml")
    with open(config_path, "r") as f:
        return load(f, Loader=FullLoader)

def get_workspaces():
    '''Returns all directory names in w_config["dir"]["w_spaces"].'''
    space_dir = os.path.join(ROOT, w_config["dir"]["w_spaces"])
    w_files = os.listdir(space_dir)
    space_names = []
    for f in w_files:
        if os.path.isdir(os.path.join(space_dir, f)):
            space_names.append(f)
    return space_names, space_dir
    
def get_model_workspaces():
    '''Retrun all model directories in the specified workspace.
    This basically returns all folders except the plot directory.
    
    Note: A Model workspace describes a directory inside the WORKSPACE. Every time a active model
    is selected a model workspace will be created to encapsulate the results of different models.
    '''
    files = os.listdir(WORKSPACE)
    names = []
    for name in files: 
        if "plot" in name:
            continue
        try: # find all dirs that names are numbers only => model directory
            # int(name)
            names.append(name)
        except Exception: 
            pass
    return names

def create_and_deploy_linkfile(rel_img_dir, rel_gt_dir, link_name, deploy_path):
    '''Create paths according to given relative paths to the image and gt directory.
    Subsequent create linkfile and store it in workspace.
    Finally copy created linkfile to deploy_path.

    '''
    print("Try to link files from {}".format(link_name))
    img_dir = os.path.join(ROOT, rel_img_dir)
    if not os.path.exists(img_dir):
        raise Exception("Img dir: {} was not found!".format(img_dir))
    gt_dir = os.path.join(ROOT, rel_gt_dir)
    if not os.path.exists(gt_dir):
        raise Exception("Gt dir: {} was not found!".format(gt_dir))
    link_name_path = os.path.join(WORKSPACE, link_name + ".txt")
    sample_count = create_link_file(img_dir, gt_dir, link_name_path)
    copyfile(link_name_path, deploy_path)
    return sample_count


def gen_linkfiles(prep_train):
    '''Generates linkfiles for train and validation set of given dataset.
    Both files are copied in the workspace and also copied/deployed in the imagemap directory
    Retruns amount of train and validation samples.
    '''
    if prep_train in d_config["dataset"].keys():
        # link dataset files
        # train data
        train_deploy = os.path.join(ROOT, w_config["path"]["linkfile_train_deploy"])
        val_deploy = os.path.join(ROOT, w_config["path"]["linkfile_val_deploy"])
        train_sample_count = create_and_deploy_linkfile(d_config["dataset"][prep_train]["train"]["img"], 
                                    d_config["dataset"][prep_train]["train"]["gt"], 
                                    prep_train+"_train", 
                                    train_deploy)
        val_sample_count = create_and_deploy_linkfile(d_config["dataset"][prep_train]["val"]["img"], 
                                    d_config["dataset"][prep_train]["val"]["gt"], 
                                    prep_train+"_val", 
                                    val_deploy)
        return train_sample_count, val_sample_count
    else:
        raise Exception("Could not assign data to {}!\Do you have a dataset definition for {} in your gpu.yml?".format(prep_train, prep_train))
    


def segment_dataset(segment, active_model):
    if segment in d_config["dataset"].keys():
        # active_model = load_config()[w_config["sel_model"]]
        test_img_dir = d_config["dataset"][segment]["train" if train_mode else "val"]["img"] # load image dir
        model_path = get_workspace_model()
        if model_path:
            deploy_path = w_config["path"]["proto_deploy"] # create prototxt path
            out_dir = os.path.join( WORKSPACE, active_model, d_config["dataset"][segment]["train" if train_mode else "val"]["out"]) 
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            seg_images(test_img_dir, out_dir, deploy_path, model_path, link_file=w_config["files"]["data"])
    else:
        raise Exception("Flag {} does not link to a Dataset.\n Is it defined in gpu.yml?".format(segment))


def calculate_metrics(metrics, active_model):
    # active_model = load_config()[w_config["sel_model"]]

    pred_dir = os.path.join(WORKSPACE, active_model, d_config["dataset"][metrics]["train" if train_mode else "val"]["out"])
    gt_dir = os.path.join(ROOT, d_config["dataset"][metrics]["train" if train_mode else "val"]["gt"])
    if (not os.path.exists(pred_dir)) :
        raise Exception("Could not find prediction data.\nDid you run a segmentation?")
    if (not os.path.exists(gt_dir)):
        raise Exception(
            "Could not find groud truth data.\nDid you add the correct path to the gpu.yml?")
    metric_path = os.path.join(WORKSPACE, active_model, d_config["dataset"][metrics]["train" if train_mode else "val"]["out"], 
                w_config["files"]["metric"])
    exe_dir = os.path.join(ROOT, w_config["dir"]["metric_build"])
    if (not os.path.exists(exe_dir)):
        raise Exception("Could not find Metrics build dir.\nDid you build it correctly?")
    os.chdir(exe_dir)  # switch dir to the executabel
    cal_metrics(pred_dir, gt_dir, metric_path)
    os.chdir(ROOT)


def save_incremental_plot(axs, out_path):
    '''Create legend for gven axes and save the plot.
    Paramerters:

    axs - list of two matplotlib axes
    out_path - path/name of the output plot
    '''
    box = axs[0].get_position()
    axs[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    box = axs[1].get_position()
    axs[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    eva, select, segment, metrics, link_files, gen_plots, prep_train, train_mode, seg_all, cal_all_metrics, gen_all_plots, list_workspaces, col_plots, gen_pr_roc_plot, retrain, plot_models = parser()
    # Ensure that the filesystem is prepared and no dir is missing:

    if not os.path.exists(os.path.join(ROOT, w_config["dir"]["snapshot"])):
        os.makedirs(os.path.join(ROOT, w_config["dir"]["snapshot"]))
    if not os.path.exists(os.path.join(ROOT, w_config["dir"]["logs"])):
        os.makedirs(os.path.join(ROOT, w_config["dir"]["logs"]))
    

    if eva:
        from scripts.eval.parse_caffe_log import parse_log
        from scripts.eval.evaluate_loss import det_best_model
        get_best_model(vis=True)
    
    if select > 0:
        copy_model(select)

    try:
        get_workspace_model()
        active_model = load_config()[w_config["w_conf"]["sel_model"]]
    except Exception:
        print("Selecting a model after training with --sel <iteration> or copy it in the workspace.")
    
    if segment:
        from scripts.segment.segment_images import seg_images
        segment_dataset(segment, active_model)

    if metrics:
        
        from scripts.eval.calMetrics import cal_metrics
        if metrics in d_config["dataset"].keys():
            calculate_metrics(metrics, active_model)
        else:
            raise Exception("Flag {} does not link to a path.\nIs it defined in workspace.yml?".format(metrics))
    # print(config)
    

    if len(link_files) == 3:
        link_name = link_files[0]
        link_name_path = os.path.join(WORKSPACE, link_name)
        img_rel_dir = link_files[1]
        img_dir = os.path.join(ROOT, split_path_at(w_config["dir"]["datasets"], img_rel_dir))
        if not os.path.exists(img_dir):
            raise Exception("Img dir: {} was not found!".format(img_dir))
        gt_rel_dir = link_files[2]
        gt_dir = os.path.join(ROOT, split_path_at(w_config["dir"]["datasets"], gt_rel_dir))     
        if not os.path.exists(gt_dir):
            raise Exception("Gt dir: {} was not found!".format(gt_dir))
        create_link_file(img_dir, gt_dir, link_name_path)
    elif (len(link_files) == 1) and (link_files[0] in d_config["dataset"].keys()):
        print("Try to link files from {}".format(link_files[0]))
        img_dir = os.path.join(ROOT, d_config["dataset"][link_files[0]]["train" if train_mode else "val"]["img"])
        if not os.path.exists(img_dir):
            raise Exception("Img dir: {} was not found!".format(img_dir))
        gt_dir = os.path.join(ROOT, d_config["dataset"][link_files[0]]["train" if train_mode else "val"]["gt"])
        if not os.path.exists(gt_dir):
            raise Exception("Gt dir: {} was not found!".format(gt_dir))
        link_name_path = os.path.join(WORKSPACE, link_files[0]+".txt")
        create_link_file(img_dir, gt_dir, link_name_path)
    # else:
    #     raise Exception("Invalid count of arguments: {}".format(len(link_files)))

    if gen_plots:
        from scripts.eval.plots.plot_metrics import create_plots
        if gen_plots in d_config["dataset"].keys():
            metric_path = os.path.join(WORKSPACE, d_config["dataset"][gen_plots]["train" if train_mode else "val"]["out"] , w_config["files"]["metric"])
            if not os.path.exists(metric_path):
                raise Exception("Could not find metric path!\nDid you run --cal-metrics before?")
            plot_dir = os.path.join(WORKSPACE, d_config["dataset"][gen_plots]["train" if train_mode else "val"]["out"], w_config["plotdir"])
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            create_plots(metric_path, plot_dir)
        else:
            raise Exception("Flag {} does not link to a path.\nIs it defined in workspace.yml?".format(gen_plots))

    if prep_train:
        train_samples, val_samples = gen_linkfiles(prep_train)
        from scripts.model.edit_training import rename_train_cond, set_net_proto_paths, set_model_config
        model_config_path = os.path.join(ROOT, w_config["path"]["m_config"])
        set_model_config(model_config_path, 
            solver_path=os.path.join(ROOT, w_config["path"]["proto_solver"]), 
            test_iter=val_samples,
            retrain=False, weight_path=os.path.join(ROOT, w_config["path"]["base_weights"]))
        out_dir = os.path.join(ROOT, w_config["dir"]["model"])
        rename_train_cond(out_dir, prep_train)
        set_net_proto_paths(ROOT, os.path.join(ROOT, w_config["path"]["linkfile_train_deploy"]), w_config["path"]["proto_train"])
        set_net_proto_paths(ROOT, os.path.join(ROOT, w_config["path"]["linkfile_val_deploy"]), w_config["path"]["proto_val"])
        store_in_config({w_config["w_conf"]["model_prefix"]: prep_train})

    if seg_all:
        from scripts.segment.segment_images import seg_images
        for segment in d_config["dataset"].keys():
            segment_dataset(segment, active_model)

    
    if cal_all_metrics:
        from scripts.eval.calMetrics import cal_metrics
        for metrics in d_config["dataset"].keys():
            calculate_metrics(metrics, active_model)
    # print(config)


    if list_workspaces:
        names, space_dir = get_workspaces()
        print("Existing Workspaces: {}".format(", ".join(names)))


    if retrain:
        from scripts.model.edit_training import set_net_proto_paths, rename_train_cond
        if retrain in d_config["dataset"].keys():
            model_path = get_workspace_model() # get abs model path
            # load config and adapt model path
            config_path = os.path.join(ROOT, w_config["path"]["m_config"])
            #TODO Adapt learn rate in trainer
            workspace_name = w_config["w_dir"]["current_space"] + "_on_" + retrain
            # create new workspace 
            os.chdir(os.path.join(ROOT, w_config["dir"]["scripts"])) # Change dir to find workspace config
            create_workspace(switch_to=workspace_name)
            os.chdir(ROOT)
            store_in_config({w_config["w_conf"]["model_prefix"]: workspace_name})
            # create linkfiles
            train_samples, val_samples = gen_linkfiles(retrain)
            # Set the path to linkfiles in prototxt files
            set_net_proto_paths(ROOT, os.path.join(ROOT, w_config["path"]["linkfile_train_deploy"]), w_config["path"]["proto_train"])
            set_net_proto_paths(ROOT, os.path.join(ROOT, w_config["path"]["linkfile_val_deploy"]), w_config["path"]["proto_val"])
            # Store workspace_name in workspace config file
            out_dir = os.path.join(ROOT, w_config["dir"]["model"])
            rename_train_cond(out_dir, workspace_name)
            if not os.path.exists(config_path):
                raise Exception("Config file does not exist!")

            # Write changes to model config
            with open(config_path, "r") as f:
                m_conf = load(f, Loader=FullLoader)
                m_conf["weights"]["retrain"] = model_path
                m_conf["retrain"] = True
            with open(config_path, "w") as f:
                dump(m_conf, f)
            # 
        else:
            raise Exception("Dataset: {} not found!".format(retrain))
        
    if plot_models:
        import matplotlib.pyplot as plt 
        from scripts.eval.plots.plot_metrics import create_incremental_plot
        files = os.listdir(WORKSPACE)
        model_dir_names = get_model_workspaces()
        print(model_dir_names)
        dataset_type = "train" if train_mode else "val"
        plot_dst_dir = os.path.join(WORKSPACE, w_config["plotdir"])
        if not os.path.exists(plot_dst_dir):
            os.makedirs(plot_dst_dir)
        for d_set in d_config["dataset"].keys():
            fig, axs = plt.subplots(2,1, figsize=(15,15))
            for model_name in model_dir_names:
                metric_path = os.path.join(WORKSPACE, model_name, d_config["dataset"][d_set]["train" if train_mode else "val"]["out"] , w_config["files"]["metric"])
                if not os.path.exists(metric_path):
                    continue
                create_incremental_plot(metric_path, axs, model_name)  
            out_path = os.path.join(plot_dst_dir, "_".join([d_set, dataset_type])+".pdf")
            save_incremental_plot(axs, out_path)


    if gen_pr_roc_plot:
        raise NotImplementedError()
    #     import matplotlib.pyplot as plt 
    #     from scripts.eval.plots.plot_metrics import create_incremental_plot
        
    #     names, space_dir = get_workspaces()
    #     print(names)
    #     dataset_type = "train" if train_mode else "val"
    #     plot_dst_dir = os.path.join(ROOT, config["out_plots"])
    #     print(plot_dst_dir)
    #     if not os.path.exists(plot_dst_dir):
    #         os.makedirs(plot_dst_dir)
    #     for gen_plots in d_config["dataset"].keys():
    #         fig, axs = plt.subplots(2,1, figsize=(15,15))
    #         for work_name in names:
    #             print(gen_plots)
    #             work_dir = os.path.join(space_dir, work_name)
    #             plot_dir = os.path.join(work_dir, d_config["dataset"][gen_plots][dataset_type]["out"], w_config["plotdir"])
    #             metric_path = os.path.join(work_dir, d_config["dataset"][gen_plots]["train" if train_mode else "val"]["out"] , w_config["files"]["metric"])
    #             if not os.path.exists(metric_path):
    #                 raise Exception("Could not find metric path!\nDid you run --cal-metrics before?")
    #             if not os.path.exists(plot_dir):
    #                 os.makedirs(plot_dir)
    #             create_incremental_plot(metric_path, axs, work_name)
    #         out_path = os.path.join(ROOT, config["out_plots"], "_".join([gen_plots, dataset_type])+".pdf")
    #         save_incremental_plot(axs, out_path)

    if col_plots:
        raise NotImplementedError()
        # names, space_dir = get_workspaces()
        # print(names)
        # dataset_type = "train" if train_mode else "val"
        # plot_dst_dir = os.path.join(ROOT, w_config["w_dir"]["out_plots"])
        # print(plot_dst_dir)
        # if not os.path.exists(plot_dst_dir):
        #     os.makedirs(plot_dst_dir)
        # for work_name in names:
        #     work_dir = os.path.join(space_dir, work_name)
        #     plot_dir = os.path.join(work_dir, w_config["w_dir"]["plot"])
        #     if os.path.exists(plot_dir) and os.path.isdir(plot_dir):
        #         dst_path = os.path.join(plot_dst_dir, work_name)
        #         copytree(plot_dir, dst_path)
        #     else:
        #         print("No plots found in workspace: {}".format(work_name))

    if gen_all_plots:
        raise NotImplementedError()
        # from scripts.eval.plots.plot_metrics import create_plots
        # for gen_plots in d_config["dataset"].keys():
        #     metric_path = os.path.join(WORKSPACE, d_config["dataset"][gen_plots]["train" if train_mode else "val"]["out"] , w_config["files"]["metric"])
        #     if not os.path.exists(metric_path):
        #         raise Exception("Could not find metric path!\nDid you run --cal-metrics before?")
        #     plot_dir = os.path.join(WORKSPACE, d_config["dataset"][gen_plots]["train" if train_mode else "val"]["out"], w_config["plotdir"])
        #     if not os.path.exists(plot_dir):
        #         os.makedirs(plot_dir)
        #     create_plots(metric_path, plot_dir)
