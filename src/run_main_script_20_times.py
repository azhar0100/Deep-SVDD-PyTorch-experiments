


from main import main
import sh

try:
	sh.mkdir("../log")  
except:
	pass
try:
	sh.mkdir("../log/mnist_test")
except:
	pass
try:
	sh.mkdir("../data")
except:
	pass

import pathlib
command_str = "toydataset toynet ../log/test{0}/ ../data --objective one-class --lr 0.0001 --n_epochs 10000 --lr_milestone 50 --batch_size 150 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 10000 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 0 --retrain_decoder"

for i in range(20):
	pathlib.Path('../log/test{}'.format(str(i).zfill(2))).mkdir(parents=True, exist_ok=True)
	# for j in range(10):
		# print(i)
	# main(command_str.format(str(i).zfill(2)).split())
	print("!python src/main.py {}".format(command_str.format(str(i).zfill(2))))
