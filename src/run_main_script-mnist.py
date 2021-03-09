from main import main
import sh

try:
	sh.mkdir("../log")  
except:
	pass
try:
	sh.mkdir("../log/mnist")
except:
	pass
try:
	sh.mkdir("../data")
except:
	pass

command_str = "python main.py mnist mnist_LeNet ../log/mnist_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 3"
main(command_str.split())
