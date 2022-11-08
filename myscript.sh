#!/bin/bash

#Date: 16 September 2022
#Author: Asher Moncho

#The purpose of the script is to automate to the runnning of the network 
#algorithms that generate depth maps by accessing the environments and 
#executing the python scripts of these environments.

instructions() {
	echo
	echo "Choose a network to run"
	echo "1) adabins"
	echo "2) adelai"
	echo "3) bts"
	echo "4) densedepth"
	echo "5) dpt"
	echo "6) lapdepth"
}

#This is the code for the execution of the environments that will occur and their associated algorithms
adabins() {
	dir_path="$ENV_PATH/adabins_env"
	#conda init bash
	source ~/.bashrc
	conda deactivate
	conda activate adabins_env
	cd ~/algorithms/AdaBins
	python trial.py
}


adelai() {
	dir_path="$ENV_PATH/adelai_env"
	#conda init bash
	source ~/.bashrc
	conda deactivate
	conda activate adelai_env
	cd ~/algorithms/AdelaiDepth/LeReS/Minist_Test
	export PYTHONPATH="/home/asher/algorithms/AdelaiDepth/LeReS/Minist_Test"

	echo "Choose an architecture"
	select resnet in "res50" "res101"
	do
		case $resnet in
			"res50") python ./tools/test_depth.py --load_ckpt res50.pth --backbone resnet50; break;;
			"res101") python ./tools/test_depth.py --load_ckpt res101.pth --backbone resnext101; break;;
			*) break;;
		esac
	done
}


bts() {
	dir_path="$ENV_PATH/bts_env"
        #conda init bash
        source ~/.bashrc
        conda deactivate
        conda activate bts_env
	cd ~/workspace/bts/pytorch

	python bts_test.py arguments_test_nyu.txt


}


densedepth() {
	dir_path="$ENV_PATH/densedepth_env"
        #conda init bash
        source ~/.bashrc
        conda deactivate
        conda activate densedepth_env
	cd ~/algorithms/DenseDepth

	python test.py

}


dpt() {
	dir_path="$ENV_PATH/dpt_env"
        #conda init bash
        source ~/.bashrc
        conda deactivate
        conda activate dpt_env

}


lapdepth() {
	dir_path="$ENV_PATH/lapdeth_env"
        #conda init bash
        source ~/.bashrc
        conda deactivate
        conda activate lapdepth_env
	cd ~/algorithms/LapDepth-release

	select pretrained in "KITTI" "NYU"
	do
		case $pretrained in
			"KITTI") python demo.py --model_dir ./pretrained/LDRN_KITTI_ResNext101_pretrained_data.pkl --img_dir ~/algorithms/LapDepth-release/example/kitti_demo.jpg --pretrained KITTI --cuda --gpu_num 0; break;;
			"NYU") python demo.py --model_dir ./pretrained/LDRN_NYU_ResNext101_pretrained_data.pkl --img_dir ~/algorithms/LapDepth-release/example/nyu_demo.jpg --pretrained NYU --cuda --gpu_num 0; break;;
			*) break;;
		esac
	done
}

vnl() {

	dir_path="$ENV_PATH/vnl_env"
        #conda init bash
        source ~/.bashrc
        conda deactivate
        conda activate vnl_env
	cd ~/algorithms/VNL_Monocular_Depth_Prediction

	select pretrained in "Inference of NYUDV2 dataset" "Test Prediction"
	do
		case $pretrained in
			"Inference on NYUDV2 dataset") python ./tools/test_nyu_metric.py \ --dataroot ; break;;
			"Test Prediction")  break;;
			*) break;;
		esac
	done

}


#This is the code to select which algorithm to run based on an input image
choosealgorithm() {

	select algorithm in "adabins" "adelai" "bts" "densedepth" "dpt" "lapdepth"
	do
		case $algorithm in
			"adabins") adabins; instructions;;
			"adelai") adelai; instructions;;
			"bts") bts; instructions;;
			"densedepth") densedepth; instructions;;
			"dpt") dpt; instructions;;
			"lapdepth") lapdepth; instructions;;
		esac
	done

}

source ~/anaconda3/etc/profile.d/conda.sh

ENV_PATH="$HOME/anaconda3/envs"

echo "Choose a network to run"

#echo "The following environments are installed: $(ls $ENV_PATH)"

case $1 in
	"adabins") adabins; instructions;;
	"adelai") adelai; instructions;;
	"bts") bts; instructions;;
	"densedepth") densedepth; instructions;;
	"dpt") dpt; instructions;;
	"lapdepth") lapdepth; instructions;;
	*) choosealgorithm;;
esac

