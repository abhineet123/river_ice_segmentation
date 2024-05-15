%% important: remember to run the code from the directory where main.m is located
clc
clear all
close all
%%
db_root_dir='/data/617/images';
vid_name='YUN00001_1280x720_1_120';
%% params
exclude_boundaries=1; %0 when moving object is close to boundary at many frames
remove_low_energy_blobs=1; % 0 when you want to keep also the small objects
optical_flow_method='CeLiu';%'Brox';%;
thr=0.2;% threshold to get binary segmentation
%% flags
all_ready=0; % put 1 if you want only to create the movies
DB_ready=0;
prepare_SP=0;
compute_MS=0;
phase1_done=0;
phase2_done=0;
%% set pathes 
base_dir=cd;
addpath(genpath(base_dir));
base_results_dirname=fullfile(base_dir,'data','results',vid_name);
%%
if ~all_ready
    if ~DB_ready
        load_vid(db_root_dir,vid_name);
    end
    vid=importdata(fullfile(base_results_dirname,'vid_MS.mat'));
    vid2=importdata(fullfile(base_results_dirname,'vid_org.mat'));
    vid3=importdata(fullfile(base_results_dirname,'vid3_LAB.mat'));
    if ~prepare_SP
        get_SP_data(vid2,vid3,1,base_dir,vid_name,'LR');
        SP_hist=importdata(fullfile(base_results_dirname,'SP_hist_LR.mat'));
        if ~exist(fullfile(base_results_dirname,['NN_data_LR.mat']))
            build_SP_graph(SP_hist,1,4,10,4,base_dir,vid_name,'LR');
        end
    end
    if ~compute_MS
        compute_saliency(vid,vid2,vid3,exclude_boundaries,base_dir,vid_name,optical_flow_method);
    end
    if ~phase1_done
        perform_phase1(base_dir,vid_name,remove_low_energy_blobs)
    end
    if ~phase2_done
        perform_phase2(vid2,vid3,base_dir,vid_name);
    end
end
%%
MS=importdata(fullfile(base_results_dirname,'MS.mat'));
save_saliency_vid( vid2,MS,base_results_dirname,vid_name,'saliency');
MS=importdata(fullfile(base_results_dirname,'MS_phase1.mat'));
save_seg_vid( vid2,MS,thr,base_results_dirname,vid_name,'phase1');
MS=importdata(fullfile(base_results_dirname,'MS_phase2.mat'));
save_seg_vid( vid2,MS,thr,base_results_dirname,vid_name,'phase2');