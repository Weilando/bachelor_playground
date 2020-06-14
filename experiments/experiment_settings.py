def get_settings_lenet_mnist():
    experiment_settings = dict()
    experiment_settings['net_count'] = 1
    experiment_settings['plan_fc'] = [300, 100]
    experiment_settings['epoch_count'] = 2
    experiment_settings['learning_rate'] = 1.2e-3 # page 3, figure 2
    experiment_settings['prune_rate_fc'] = 0.2 # page 3, figure 2
    experiment_settings['prune_count'] = 1
    experiment_settings['prune_method'] = 'imp'
    experiment_settings['loss_plot_step'] = 100
    experiment_settings['net'] = 'lenet'
    experiment_settings['dataset'] = 'mnist'
    return experiment_settings

def get_settings_conv2_cifar10():
    experiment_settings = dict()
    experiment_settings['net_count'] = 1
    experiment_settings['plan_conv'] = [64, 64, 'M']
    experiment_settings['plan_fc'] = [256, 256]
    experiment_settings['epoch_count'] = 2
    experiment_settings['learning_rate'] = 2e-4 # page 3, figure 2
    experiment_settings['prune_rate_conv'] = 0.1 # page 3, figure 2
    experiment_settings['prune_rate_fc'] = 0.2 # page 3, figure 2
    experiment_settings['prune_count'] = 1
    experiment_settings['prune_method'] = 'imp'
    experiment_settings['loss_plot_step'] = 100
    experiment_settings['net'] = 'conv'
    experiment_settings['dataset'] = 'cifar'
    return experiment_settings

def get_settings_conv4_cifar10():
    experiment_settings = dict()
    experiment_settings['net_count'] = 1
    experiment_settings['plan_conv'] = [64, 64, 'M', 128, 128, 'M']
    experiment_settings['plan_fc'] = [256, 256]
    experiment_settings['epoch_count'] = 2
    experiment_settings['learning_rate'] = 2e-4 # page 3, figure 2
    experiment_settings['prune_rate_conv'] = 0.1 # page 3, figure 2
    experiment_settings['prune_rate_fc'] = 0.2 # page 3, figure 2
    experiment_settings['prune_count'] = 1
    experiment_settings['prune_method'] = 'imp'
    experiment_settings['loss_plot_step'] = 100
    experiment_settings['net'] = 'conv'
    experiment_settings['dataset'] = 'cifar'
    return experiment_settings

def get_settings_conv6_cifar10():
    experiment_settings = dict()
    experiment_settings['net_count'] = 1
    experiment_settings['plan_conv'] = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']
    experiment_settings['plan_fc'] = [256, 256]
    experiment_settings['epoch_count'] = 2
    experiment_settings['learning_rate'] = 2e-4 # page 3, figure 2
    experiment_settings['prune_rate_conv'] = 0.1 # page 3, figure 2
    experiment_settings['prune_rate_fc'] = 0.2 # page 3, figure 2
    experiment_settings['prune_count'] = 1
    experiment_settings['prune_method'] = 'imp'
    experiment_settings['loss_plot_step'] = 100
    experiment_settings['net'] = 'conv'
    experiment_settings['dataset'] = 'cifar'
    return experiment_settings
