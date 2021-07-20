import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', size=16, family='serif')

names = {
    'rgf': 'RGF',
    'prgf': 'PRGF',
    'ars': 'ARS',
    'wrong_pars': 'PARS-Naive',
    'x_pars_multi': 'PARS'
}

for f_type in ['orig', 'quad', 'valley']:
    if f_type == 'orig':
        length = 200
    elif f_type == 'quad':
        length = 100
    elif f_type == 'valley':
        length = 500
    plt.figure()
    for optim_type in ['rgf', 'prgf', 'ars', 'wrong_pars', 'x_pars_multi']:
        log_errs_list = []
        for ite in range(5):
            log_errs_list.append(np.log10(np.load('exp_pert/{}_{}_{}.npy'.format(f_type, optim_type, ite))[:length]))
        log_errs_list = np.array(log_errs_list)
        log_errs_list = np.concatenate([np.zeros([5, 1]), log_errs_list], axis=1)
        # print(log_errs_list[:, -1])
        mean_log_errs = np.mean(log_errs_list, axis=0)
        std_log_errs = np.std(log_errs_list, axis=0, ddof=1)
        if optim_type[0] == 'p' or 'pars' in optim_type:
            linestyle = '-'
        else:
            linestyle = '--'
        plt.plot(np.arange(length+1), mean_log_errs, label=names[optim_type], linestyle=linestyle)
        plt.fill_between(np.arange(length+1), mean_log_errs-1.96*std_log_errs, mean_log_errs+1.96*std_log_errs, alpha=0.1)
    plt.grid()
    plt.legend(fontsize=14)
    plt.savefig('exp_pert/{}.pdf'.format(f_type), bbox_inches='tight')
