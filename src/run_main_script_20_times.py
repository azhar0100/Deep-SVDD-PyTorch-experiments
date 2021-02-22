


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

command_str = "toydataset toynet ../log/test{0} ../data --objective one-class --lr 0.0001 --n_epochs 1000 --lr_milestone 50 --batch_size 150 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 1000 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 0 --retrain_decoder --model_name model{0}.tar --config_name config{0}.json --results_name result{0}.json"
for i in range(20):
	try:
		sh.mkdir("../log/test{}".format(str(i).zfill(2)))
	except:
		pass
	main(command_str.format(str(i).zfill(2)).split())
