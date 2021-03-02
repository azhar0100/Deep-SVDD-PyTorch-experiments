from main import main
import sh

try:
	sh.mkdir("../log")  
except:
	pass
try:
	sh.mkdir("../log/toy")
except:
	pass
try:
	sh.mkdir("../data")
except:
	pass

command_str = "toydataset toynet ../log/toy ../data --objective one-class --lr 0.0001 --n_epochs 100 --lr_milestone 50 --batch_size 150 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 100 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 0 --retrain_decoder"
main(command_str.split())
