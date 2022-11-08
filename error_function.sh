#!/usr/bin/env bash

# Calculate the error metrics between the ground truths and the input image

ground_truth=''
input_image=''

adabin() {
	python error_functions.py -grt $ground_truth -ii $input_image -a AdaBins
}

adelai() {
	python error_functions.py -grt $ground_truth -ii $input_image -a AdelaiDepth
}

bts() {
	python error_functions.py -grt $ground_truth -ii $input_image -a bts
}

densedepth() {
	python error_functions.py -grt $ground_truth -ii $input_image -a DenseDepth
}

dpt() {
	python error_functions.py -grt $ground_truth -ii $input_iage -a DPT
}

lapdepth() {
	python error_functions.py -grt $ground_truth -ii $input_image -a LapDepth-release
}

vnl() {
	python error_functions.py -grt $ground_truth -ii $input_image -a VNL_Monocular_Depth_Prediction
}

echo "Choose the network to compute the error metrics"

select algorithm_error in "AdaBin" "Adelai" "bts" "DenseDepth" "DPT" "LapDepth" "VNL"
do
	case $algorithm_error in
		"AdaBins") adabins;;
		"Adelai") adelai;;
		"bts") bts;;
		"DenseDepth") densedepth;;
		"DPT") dpt;;
		"LapDepth") lapdepth;;
		"VNL") vnl;;
	esac
done
