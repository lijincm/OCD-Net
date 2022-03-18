import os

def conduct_exp(buffer):
    the_command = 'python3 main.py' \
        + ' --load_best_args'   \
        + ' --buffer_size ' + str(buffer) \
        + ' --model ocdnet'\
        + ' --dataset seq-cifar10'\
        + ' --csv_log'

    os.system(the_command)


for i in range(0,1):
    conduct_exp(200)
    #conduct_exp(500)
    #conduct_exp(5120)



