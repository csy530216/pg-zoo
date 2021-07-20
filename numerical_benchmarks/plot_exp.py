import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', size=16, family='serif')

names = {
    'rgf': 'RGF',
    'rgf-lr25': 'RGF-0.04',
    'prgf': 'PRGF',
    'prgf-lr25': 'PRGF-0.04',
    'prgf-lr50': 'PRGF-0.02',
    'ars': 'ARS',
    'ars-lr25': 'ARS-0.04',
    'pars': 'PARS',
    'pars-lr25': 'PARS-0.04',
    'pars-lr50': 'PARS-0.02'
}

for f_type in ['orig', 'quad', 'quad_area']:
    if f_type == 'orig':
        length = 500
    elif f_type == 'quad':
        length = 200
    elif f_type == 'quad_area':
        length = 200
    for draw_type in ['rgf', 'ars']:
        plt.figure()
        if draw_type == 'rgf':
            if f_type != 'quad_area':
                optim_type_list = ['rgf', 'rgf-lr25', 'prgf', 'prgf-lr25', 'prgf-lr50']
            else:
                optim_type_list = ['rgf', 'prgf']
        elif draw_type == 'ars':
            if f_type != 'quad_area':
                optim_type_list = ['ars', 'ars-lr25', 'pars', 'pars-lr25', 'pars-lr50']
            else:
                optim_type_list = ['ars', 'pars']
        for optim_type in optim_type_list:
            log_errs_list = []
            for ite in range(5):
                log_errs_list.append(np.log10(np.load('exp/{}_{}_{}.npy'.format(f_type, optim_type, ite))[:length]))
            log_errs_list = np.array(log_errs_list)
            log_errs_list = np.concatenate([np.zeros([5, 1]), log_errs_list], axis=1)
            # print(log_errs_list[:, -1])
            mean_log_errs = np.mean(log_errs_list, axis=0)
            std_log_errs = np.std(log_errs_list, axis=0, ddof=1)
            if optim_type[0] == 'p':
                linestyle = '-'
            else:
                linestyle = '--'
            plt.plot(np.arange(length+1), mean_log_errs, label=names[optim_type], linestyle=linestyle)
            plt.fill_between(np.arange(length+1), mean_log_errs-1.96*std_log_errs, mean_log_errs+1.96*std_log_errs, alpha=0.1)
        plt.grid()
        plt.legend(fontsize=14)
        plt.savefig('exp/{}_{}.pdf'.format(f_type, draw_type), bbox_inches='tight')
