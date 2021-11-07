import numpy as np
np.set_printoptions(precision=4)

def normalize(x):
    return x / (np.linalg.norm(x) + 1e-12)

def set_function(f_type):
    if f_type == 'orig':
        n = 256
        x_OPT = np.arange(n, 0, -1) / (n + 1)
        L1 = 4
        x = np.zeros(n)
        current_row = 1
        base = 2
        tau = 0

        def f(x):
            return 0.5 * x[0] ** 2 + 0.5 * np.sum(np.square((x - np.roll(x, -1))[:-1])) + 0.5 * x[-1] ** 2 - x[0]

        def grad(x):
            grad = np.zeros_like(x)
            grad[0] = 2 * x[0] - x[1] - 1
            grad[1:-1] = 2 * x[1:-1] - x[:-2] - x[2:]
            grad[-1] = 2 * x[-1] - x[-2]
            return grad

        OPT = f(x_OPT)
        S = f(x) - OPT
    elif f_type == 'quad':
        n = 256
        step = 1
        spectrum = (np.arange((n-1) * step + 1, 1-1e-10, -step) / n / step)
        # spectrum = n / (2.0 ** np.arange(n))
        # spectrum = np.ones(n)
        # spectrum[0] = 2560
        # spectrum = np.ones(n) * 256
        # spectrum[-1] = 1
        L1 = 2 * spectrum[0]
        x_OPT = np.zeros(n)
        x = np.zeros(n)
        x[-1] = n
        # x = np.ones(n)
        current_row = 1
        base = 2
        tau = 2 * spectrum[-1]

        def f(x):
            return np.sum(spectrum * x * x)

        def f_inv(y):
            return np.sqrt(y / spectrum)

        def grad(x):
            return 2 * spectrum * x

        OPT = f(x_OPT)
        S = f(x) - OPT
    elif f_type == 'valley':
        n = 256
        x_OPT = np.ones(n)
        x = np.zeros(n)
        current_row = 1
        base = 1.02
        L1 = -1
        tau = 0

        def f(x):
            return np.sum(100 * np.square((np.square(x[:-1]) - x[1:])) + np.square(x[:-1] - 1))

        def grad(x):
            ans = np.zeros(n)
            ans[0] = 400 * x[0] * (np.square(x[0]) - x[1]) + 2 * (x[0] - 1)
            ans[1:-1] = 400 * x[1:-1] * (np.square(x[1:-1]) - x[2:]) + 2 * (x[1:-1] - 1) - 200 * (np.square(x[:-2]) - x[1:-1])
            ans[-1] = -200 * (np.square(x[-2]) - x[-1])
            return ans

        OPT = f(x_OPT)
        S = f(x) - OPT

    return f, grad, x, n, L1, tau, OPT, S, current_row, base

class Avg:
    def __init__(self, direct=False):
        self.list = []
        self.direct = direct

    def update(self, value):
        if self.direct:
            self.list.append(value)
        else:
            self.list.append(value ** 2)

    @property
    def avg(self):
        if self.direct:
            return np.mean(self.list)
        else:
            return np.sqrt(np.mean(self.list))

    def clear(self):
        self.list = []

def cosine_similarity(x, y):
    return np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y))


class RGF:
    def __init__(self, lr, q=1):
        self.svrg = False
        self.lr = lr
        self.q = q
        self.post_sim = Avg()

    def return_update(self, x):
        grad_x = grad(x)
        grad_x /= np.linalg.norm(grad_x)
        prior = np.random.normal(size=n)
        prior = prior / np.linalg.norm(prior)
        us = []
        for _ in range(self.q):
            u = np.random.normal(size=n)
            u -= np.dot(u, prior) * prior
            assert np.abs(np.dot(u / np.linalg.norm(u), prior)) <= 1e-4
            for old_u in us:
                u -= np.dot(u, old_u) * old_u
            u /= np.linalg.norm(u)
            us.append(u)
        g_rand = 0
        for u in us:
            derivative_rand = (f(x + mu * u) - f(x)) / mu
            g_rand += u * derivative_rand
        derivative_prior = (f(x + mu * prior) - f(x)) / mu
        g_mu = g_rand + prior * derivative_prior
        norm_g_mu = np.linalg.norm(g_mu)
        if norm_g_mu == 0:
            g_mu = np.random.normal(size=n)
            g_mu /= np.linalg.norm(g_mu)
            g_mu *= ((f(x + mu * g_mu) - f(x)) / mu)
        self.post_sim.update(np.dot(grad_x, normalize(g_mu)))
        return x - (self.lr * g_mu)

    def clear(self):
        self.post_sim.clear()

    def statistics(self):
        return "Similarity: {:.4f}".format(self.post_sim.avg)


class PRGFPrior:
    def __init__(self, lr, q=1):
        self.svrg = False
        self.lr = lr
        self.q = q

        self.post_sim = Avg()

    def return_update(self, x, prior):
        grad_x = grad(x)
        grad_x /= np.linalg.norm(grad_x)
        u = np.random.normal(size=n)
        prior = prior / np.linalg.norm(prior)
        us = []
        for _ in range(self.q):
            u = np.random.normal(size=n)
            u -= np.dot(u, prior) * prior
            assert np.abs(np.dot(u / np.linalg.norm(u), prior)) <= 1e-4
            for old_u in us:
                u -= np.dot(u, old_u) * old_u
            u /= np.linalg.norm(u)
            us.append(u)
        g_rand = 0
        for u in us:
            derivative_rand = (f(x + mu * u) - f(x)) / mu
            g_rand += u * derivative_rand
        derivative_prior = (f(x + mu * prior) - f(x)) / mu
        g_mu = g_rand + prior * derivative_prior
        norm_g_mu = np.linalg.norm(g_mu)
        if norm_g_mu == 0:
            g_mu = np.random.normal(size=n)
            g_mu /= np.linalg.norm(g_mu)
            g_mu *= ((f(x + mu * g_mu) - f(x)) / mu)
        self.post_sim.update(np.dot(grad_x, normalize(g_mu)))
        return x - (self.lr * g_mu)

    def clear(self):
        self.post_sim.clear()

    def statistics(self):
        return "Similarity: {:.4f}".format(self.post_sim.avg)

class ARS:
    def __init__(self, L, v_ini, q=1, tau=0):
        self.svrg = False
        self.L = L
        self.gamma = L
        self.v = v_ini
        self.q = q
        self.tau = tau
        self.post_sim = Avg()

    def return_update(self, x):
        theta = (self.q + 1) ** 2 / (n ** 2) / self.L
        k = theta * (self.gamma - self.tau)
        alpha = (-k + np.sqrt(k * k + 4 * theta * self.gamma)) / 2
        beta = alpha * self.gamma / (self.gamma + alpha * self.tau)
        self.gamma = (1 - alpha) * self.gamma + alpha * self.tau
        y = (1 - beta) * x + beta * self.v

        prior = np.random.normal(size=n)
        prior /= np.linalg.norm(prior)
        us = []
        for _ in range(self.q):
            u = np.random.normal(size=n)
            u -= np.dot(u, prior) * prior
            assert np.abs(np.dot(u / np.linalg.norm(u), prior)) <= 1e-4
            for old_u in us:
                u -= np.dot(u, old_u) * old_u
            u /= np.linalg.norm(u)
            us.append(u)
        g_rand = 0
        for u in us:
            derivative_rand = (f(y + mu * u) - f(y)) / mu
            g_rand += u * derivative_rand
        derivative_prior = (f(y + mu * prior) - f(y)) / mu
        g_mu = g_rand + prior * derivative_prior
        self.post_sim.update(np.dot(normalize(grad(y)), normalize(g_mu)))
        g1 = g_mu
        g2 = n / (self.q + 1) * g_mu
        lmda = alpha / self.gamma * self.tau
        self.v = (1 - lmda) * self.v + lmda * y - theta / alpha * g2

        return y - 1 / self.L * g1

    def clear(self):
        self.post_sim.clear()

    def statistics(self):
        return "Similarity: {:.4f}".format(self.post_sim.avg)


class NaivePARSPrior:
    def __init__(self, L, v_ini, q=1, tau=0):
        self.svrg = False
        self.L = L
        self.gamma = L
        self.v = v_ini
        self.q = q
        self.tau = tau
        self.post_sim = Avg()

    def return_update(self, x, prior):
        theta = (self.q + 1) ** 2 / (n ** 2) / self.L
        k = theta * (self.gamma - self.tau)
        alpha = (-k + np.sqrt(k * k + 4 * theta * self.gamma)) / 2
        beta = alpha * self.gamma / (self.gamma + alpha * self.tau)
        self.gamma = (1 - alpha) * self.gamma + alpha * self.tau
        y = (1 - beta) * x + beta * self.v

        # prior = np.random.normal(size=n)
        prior /= np.linalg.norm(prior)
        us = []
        for _ in range(self.q):
            u = np.random.normal(size=n)
            u -= np.dot(u, prior) * prior
            assert np.abs(np.dot(u / np.linalg.norm(u), prior)) <= 1e-4
            for old_u in us:
                u -= np.dot(u, old_u) * old_u
            u /= np.linalg.norm(u)
            us.append(u)
        g_rand = 0
        for u in us:
            derivative_rand = (f(y + mu * u) - f(y)) / mu
            g_rand += u * derivative_rand
        derivative_prior = (f(y + mu * prior) - f(y)) / mu
        g_mu = g_rand + prior * derivative_prior
        self.post_sim.update(np.dot(normalize(grad(y)), normalize(g_mu)))
        g1 = g_mu
        g2 = n / (self.q + 1) * g_mu
        lmda = alpha / self.gamma * self.tau
        self.v = (1 - lmda) * self.v + lmda * y - theta / alpha * g2

        return y - 1 / self.L * g1

    def clear(self):
        self.post_sim.clear()

    def statistics(self):
        return "Similarity: {:.4f}".format(self.post_sim.avg)


class PARSPrior:
    def __init__(self, L, v_ini, q=1, tau=0):
        self.svrg = False
        self.L = L
        self.gamma = L
        self.v = v_ini
        self.q = q
        self.tau = tau

        self.post_sim = Avg()
        self.mean_grad_norm2 = 1e10
        self.grad_norm2s = []

    def return_update(self, x, prior):
        # Compute try theta
        derivative_x = (f(x + mu * normalize(prior)) - f(x)) / mu
        Dt = derivative_x ** 2 / self.mean_grad_norm2
        if Dt >= 0.6:
            Dt = 0.6
        theta = (Dt + self.q / (n - 1) * (1 - Dt)) / self.L / (Dt + (n - 1) / self.q * (1 - Dt))

        k = theta * (self.gamma - self.tau)
        alpha = (-k + np.sqrt(k * k + 4 * theta * self.gamma)) / 2
        beta = alpha * self.gamma / (self.gamma + alpha * self.tau)
        y_try = (1 - beta) * x + beta * self.v

        derivative_x = (f(y_try + mu * normalize(prior)) - f(y_try)) / mu
        Dt = derivative_x ** 2 / self.mean_grad_norm2
        if Dt >= 0.6:
            Dt = 0.6
        theta = (Dt + self.q / (n - 1) * (1 - Dt)) / self.L / (Dt + (n - 1) / self.q * (1 - Dt))

        k = theta * (self.gamma - self.tau)
        alpha = (-k + np.sqrt(k * k + 4 * theta * self.gamma)) / 2
        beta = alpha * self.gamma / (self.gamma + alpha * self.tau)
        self.gamma = (1 - alpha) * self.gamma + alpha * self.tau
        y = (1 - beta) * x + beta * self.v

        prior = prior / np.linalg.norm(prior)
        us = []
        for _ in range(self.q):
            u = np.random.normal(size=n)
            u -= np.dot(u, prior) * prior
            assert np.abs(np.dot(u / np.linalg.norm(u), prior)) <= 1e-4
            for old_u in us:
                u -= np.dot(u, old_u) * old_u
            u /= np.linalg.norm(u)
            us.append(u)
        g_rand = 0
        drs = []
        for u in us:
            derivative_rand = (f(y + mu * u) - f(y)) / mu
            drs.append(derivative_rand)
            g_rand += u * derivative_rand
        derivative_prior = (f(y + mu * prior) - f(y)) / mu
        g_mu = g_rand + prior * derivative_prior

        self.post_sim.update(np.dot(normalize(grad(y)), normalize(g_mu)))
        g1 = g_mu

        norm_term = np.mean(np.square(drs))
        est_norm = norm_term * (n - 1) + derivative_prior ** 2

        self.grad_norm2s.append(est_norm)
        self.grad_norm2s = self.grad_norm2s[-10:]
        self.mean_grad_norm2 = np.mean(self.grad_norm2s)
        # The averaging trick could be omitted: we can directly set
        # ``self.mean_grad_norm2 = est_norm`` here

        g_prior = prior * derivative_prior
        g2 = (n - 1) / self.q * g_rand + g_prior

        lmda = alpha / self.gamma * self.tau
        self.v = (1 - lmda) * self.v + lmda * y - (theta / alpha * g2)
        return y - 1 / self.L * g1

    def clear(self):
        self.post_sim.clear()

    def statistics(self):
        return "Similarity: {:.4f}".format(self.post_sim.avg)


f_types = ['valley', 'quad', 'orig']
optim_types = ['rgf', 'prgf', 'ars', 'naive_pars', 'pars']

for ite in range(5):
    for f_type in f_types:
        for optim_type in optim_types:
            mu = 1e-6
            f, grad, x, n, L1, tau, OPT, S, current_row, base = set_function(f_type)
            print(ite, f_type, optim_type)
            print("mu={}, OPT={}, S={}, f(x0)-OPT={}".format(mu, OPT, S, f(x) - OPT))
            not_restart = True
            if f_type == 'quad':
                n_epochs = 100
            elif f_type == 'orig':
                n_epochs = 300
            elif f_type == 'valley':
                n_epochs = 500

            if f_type != 'valley':
                L1 *= 1
                q = 10
                if optim_type == 'rgf':
                    optim = RGF(lr=1 / L1, q=q)
                elif optim_type == 'prgf':
                    optim = PRGFPrior(lr=1 / L1, q=q)
                elif optim_type == 'ars':
                    optim = ARS(L=L1, v_ini=x, q=q, tau=tau)
                elif optim_type == 'naive_pars':
                    optim = NaivePARSPrior(L=L1, v_ini=x, q=q, tau=tau)
                elif optim_type == 'pars':
                    optim = PARSPrior(L=L1, v_ini=x, q=q, tau=tau)
                else:
                    raise Exception
            elif f_type == 'valley':
                q = 10
                # Following values of L1 are tuned
                if optim_type == 'rgf':
                    L1 = 150
                    optim = RGF(lr=1 / L1, q=q)
                elif optim_type == 'prgf':
                    L1 = 250
                    optim = PRGFPrior(lr=1 / L1, q=q)
                elif optim_type == 'ars':
                    L1 = 600
                    optim = ARS(L=L1, v_ini=x, q=q, tau=tau)
                elif optim_type == 'naive_pars':
                    L1 = 600
                    optim = NaivePARSPrior(L=L1, v_ini=x, q=q, tau=tau)
                elif optim_type == 'pars':
                    L1 = 1000
                    optim = PARSPrior(L=L1, v_ini=x, q=q, tau=tau)
                else:
                    raise Exception
            else:
                raise Exception

            errs = []

            if 'Prior' in str(type(optim)):
                bias = np.load('bias.npy')

            for epoch in range(1, n_epochs + 1):
                if hasattr(optim, 'q'):
                    q_per = optim.q + 1
                else:
                    q_per = 2
                for i in range(n // q_per):
                    if 'Prior' not in str(type(optim)):
                        x = optim.return_update(x)
                    else:
                        true_grad = grad(x)
                        noise = np.random.normal(size=n)
                        noise /= np.linalg.norm(noise)
                        if i % 2 == 1:
                            coff = 1
                        else:
                            coff = 1
                        prior = true_grad + coff * (bias + 1.5 * noise) * np.linalg.norm(true_grad)
                        # print('ground truth', np.sum(normalize(prior) * normalize(true_grad)))
                        # prior = bias
                        x = optim.return_update(x, prior)
                if f_type != 'valley':
                    print(optim.statistics())
                optim.clear()

                err = (f(x) - OPT) / S
                errs.append(err)
                # if 'valley' not in f_type:
                    # print(err)
                while err <= base ** (-current_row):
                    print(current_row, epoch)
                    current_row += 1

            errs = np.array(errs)
            np.save('exp_pert/{}_{}_{}.npy'.format(f_type, optim_type, ite), errs)
