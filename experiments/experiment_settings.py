def get_settings_lenet_mnist():
    """ Original experiment with Lenet 300-100 on MNIST. """
    experiment_settings = dict()
    experiment_settings['net_count'] = 3
    experiment_settings['plan_fc'] = [300, 100]
    experiment_settings['epoch_count'] = 60 # 55000 iterations / 917 batches ~ 60 epochs
    experiment_settings['learning_rate'] = 1.2e-3 # page 3, figure 2
    experiment_settings['prune_rate_fc'] = 0.2 # page 3, figure 2
    experiment_settings['prune_count'] = 3
    experiment_settings['prune_method'] = 'imp'
    experiment_settings['loss_plot_step'] = 100
    experiment_settings['net'] = 'lenet'
    experiment_settings['dataset'] = 'mnist'
    return experiment_settings

def get_settings_conv2_cifar10():
    """ Original experiment with Conv-6 on CIFAR-10. """
    experiment_settings = dict()
    experiment_settings['net_count'] = 3
    experiment_settings['plan_conv'] = [64, 64, 'M']
    experiment_settings['plan_fc'] = [256, 256]
    experiment_settings['epoch_count'] = 27 # 20000 iterations / 750 batches ~ 26.6 epochs
    experiment_settings['learning_rate'] = 2e-4 # page 3, figure 2
    experiment_settings['prune_rate_conv'] = 0.1 # page 3, figure 2
    experiment_settings['prune_rate_fc'] = 0.2 # page 3, figure 2
    experiment_settings['prune_count'] = 3
    experiment_settings['prune_method'] = 'imp'
    experiment_settings['loss_plot_step'] = 100
    experiment_settings['net'] = 'conv'
    experiment_settings['dataset'] = 'cifar'
    return experiment_settings

def get_settings_conv4_cifar10():
    """ Original experiment with Conv-4 on CIFAR-10. """
    experiment_settings = dict()
    experiment_settings['net_count'] = 3
    experiment_settings['plan_conv'] = [64, 64, 'M', 128, 128, 'M']
    experiment_settings['plan_fc'] = [256, 256]
    experiment_settings['epoch_count'] = 34 # 25000 iterations / 750 batches ~ 33.3 epochs
    experiment_settings['learning_rate'] = 3e-4 # page 3, figure 2
    experiment_settings['prune_rate_conv'] = 0.1 # page 3, figure 2
    experiment_settings['prune_rate_fc'] = 0.2 # page 3, figure 2
    experiment_settings['prune_count'] = 3
    experiment_settings['prune_method'] = 'imp'
    experiment_settings['loss_plot_step'] = 100
    experiment_settings['net'] = 'conv'
    experiment_settings['dataset'] = 'cifar'
    return experiment_settings

def get_settings_conv6_cifar10():
    """ Original experiment with Conv-6 on CIFAR-10. """
    experiment_settings = dict()
    experiment_settings['net_count'] = 3
    experiment_settings['plan_conv'] = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']
    experiment_settings['plan_fc'] = [256, 256]
    experiment_settings['epoch_count'] = 40 # 30000 iterations / 750 batches = 40 epochs
    experiment_settings['learning_rate'] = 3e-4 # page 3, figure 2
    experiment_settings['prune_rate_conv'] = 0.15 # page 3, figure 2
    experiment_settings['prune_rate_fc'] = 0.2 # page 3, figure 2
    experiment_settings['prune_count'] = 3
    experiment_settings['prune_method'] = 'imp'
    experiment_settings['loss_plot_step'] = 100
    experiment_settings['net'] = 'conv'
    experiment_settings['dataset'] = 'cifar'
    return experiment_settings

def get_settings_lenet_mnist_f():
    """ Faster version of experiment with Lenet 300-100 on MNIST. """
    experiment_settings = get_settings_lenet_mnist()
    experiment_settings['epoch_count'] = 12
    return experiment_settings

def get_settings_conv2_cifar10_f():
    """ Faster version of experiment with Conv-2 on CIFAR-10. """
    experiment_settings = get_settings_conv2_cifar10()
    experiment_settings['epoch_count'] = 12
    return experiment_settings

def get_settings_conv4_cifar10_f():
    """ Faster version of experiment with Conv-4 on CIFAR-10. """
    experiment_settings = get_settings_conv4_cifar10()
    experiment_settings['epoch_count'] = 14
    experiment_settings['prune_count'] = 3
    return experiment_settings

def get_settings_conv6_cifar10_f():
    """ Faster version of experiment with Conv-6 on CIFAR-10. """
    experiment_settings = get_settings_conv6_cifar10()
    experiment_settings['epoch_count'] = 16
    return experiment_settings
